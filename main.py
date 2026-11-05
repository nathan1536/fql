import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from envs.state_utils import compute_live_state
from utils.datasets import Dataset, FrameReplayBuffer, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_float('camera_fovy_scale', None, 'Scale of the field of view of the pixel camera (occlusion test).')
flags.DEFINE_float(
    'critic_camera_fovy_scale', None, 'If given, the critic sees images with this field-of-view scale (FQL only).'
)
flags.DEFINE_integer('frames_on_disk', 0, 'Whether to store online image frames in a file instead of in memory.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    config = FLAGS.agent
    critic_observations = FLAGS.critic_camera_fovy_scale is not None
    if critic_observations:
        # Set before logging the flags, so that the logged config is the one used.
        assert config['agent_name'] == 'fql', 'Separate critic observations are only supported for FQL.'
        config['critic_observations'] = True

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    asym_critic = config.get('asym_critic', False)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=FLAGS.frame_stack,
        add_states=asym_critic,
        fovy_scale=FLAGS.camera_fovy_scale,
        critic_fovy_scale=FLAGS.critic_camera_fovy_scale,
    )
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    # Image observations are stored once per frame (see `FrameReplayBuffer`); the buffer then holds the training
    # dataset plus `online_steps` transitions, regardless of `buffer_size`.
    use_frame_buffer = train_dataset['observations'].ndim == 4
    # With `frames_on_disk`, online frames go to a file in the run directory (deleted at the end of training).
    frames_path = os.path.join(FLAGS.save_dir, 'replay_frames.dat') if FLAGS.frames_on_disk else None
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        if use_frame_buffer:
            replay_buffer = FrameReplayBuffer.create(
                FLAGS.online_steps, example_transition=example_transition, frames_path=frames_path
            )
        else:
            replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    elif FLAGS.online_steps == 0:
        # Offline RL only: sample from the training dataset directly instead of copying it into a replay buffer.
        replay_buffer = train_dataset
    elif use_frame_buffer:
        # Use the training dataset as the replay buffer.
        train_dataset = FrameReplayBuffer.create(
            FLAGS.online_steps, init_dataset=train_dataset, frames_path=frames_path
        )
        replay_buffer = train_dataset
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    create_kwargs = dict(ex_states=example_batch['states']) if asym_critic else dict()
    if critic_observations:
        create_kwargs['ex_critic_observations'] = example_batch['critic_observations']
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        **create_kwargs,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    # IFQL picks actions with its critic, which takes privileged states with `asym_critic`.
    actor_needs_states = asym_critic and config['agent_name'] == 'ifql'
    # The replay buffer stores single frames, as the datasets do; `sample` stacks them.
    frame_channels = train_dataset['observations'].shape[-1]

    def to_frame(ob):
        return ob[..., -frame_channels:] if FLAGS.frame_stack is not None else ob

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()
                if asym_critic:
                    state = compute_live_state(env)

            actor_kwargs = dict(states=state) if actor_needs_states else dict()
            action = agent.sample_actions(observations=ob, temperature=1, seed=key, **actor_kwargs)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated
            if asym_critic:
                next_state = compute_live_state(env)

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            transition = dict(
                observations=to_frame(ob),
                actions=action,
                rewards=reward,
                terminals=float(done),
                masks=1.0 - terminated,
                next_observations=to_frame(next_ob),
            )
            if asym_critic:
                transition.update(states=state, next_states=next_state)
                state = next_state
            replay_buffer.add_transition(transition)
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    if frames_path is not None and os.path.exists(frames_path):
        os.remove(frames_path)


if __name__ == '__main__':
    app.run(main)
