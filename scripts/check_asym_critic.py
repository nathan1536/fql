"""Check the `asym_critic` flag without training.

1. With states equal to the observations (no encoder), an agent with `asym_critic` must be identical to one without
   it: same parameters after the same updates.
2. With pixel observations and an encoder, the agent with `asym_critic` must run, the critic (and value) must take
   states without an encoder, and the actor must keep its encoder.

Runs on CPU in seconds:
    JAX_PLATFORMS=cpu python scripts/check_asym_critic.py
"""

import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents  # noqa: E402
from agents.fql import get_config as fql_config  # noqa: E402
from agents.ifql import get_config as ifql_config  # noqa: E402
from agents.iql import get_config as iql_config  # noqa: E402
from agents.sac import get_config as sac_config  # noqa: E402

NUM_UPDATES = 5
BATCH_SIZE = 32
OB_DIM, STATE_DIM, ACTION_DIM = 28, 28, 5


def make_batch(rng, observations_shape, observations_dtype=np.float32):
    def obs():
        if observations_dtype == np.uint8:
            return rng.integers(0, 256, (BATCH_SIZE, *observations_shape), dtype=np.uint8)
        return rng.normal(size=(BATCH_SIZE, *observations_shape)).astype(np.float32)

    return dict(
        observations=obs(),
        next_observations=obs(),
        actions=rng.uniform(-1, 1, (BATCH_SIZE, ACTION_DIM)).astype(np.float32),
        rewards=rng.normal(size=BATCH_SIZE).astype(np.float32),
        masks=np.ones(BATCH_SIZE, dtype=np.float32),
        terminals=np.zeros(BATCH_SIZE, dtype=np.float32),
        states=rng.normal(size=(BATCH_SIZE, STATE_DIM)).astype(np.float32),
        next_states=rng.normal(size=(BATCH_SIZE, STATE_DIM)).astype(np.float32),
    )


def small_config(get_config, **kwargs):
    config = get_config()
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.batch_size = BATCH_SIZE
    if 'flow_steps' in config:
        config.flow_steps = 2
    if 'num_samples' in config:
        config.num_samples = 4
    for k, v in kwargs.items():
        config[k] = v
    return config


def run(get_config, batches, **kwargs):
    config = small_config(get_config, **kwargs)
    agent = agents[config['agent_name']].create(
        0, batches[0]['observations'], batches[0]['actions'], config, ex_states=batches[0]['states']
    )
    for batch in batches:
        agent, _ = agent.update(batch)
    return agent


def check_equivalence(name, get_config):
    rng = np.random.default_rng(0)
    batches = [make_batch(rng, (OB_DIM,)) for _ in range(NUM_UPDATES)]
    for batch in batches:
        batch['states'], batch['next_states'] = batch['observations'], batch['next_observations']

    params = run(get_config, batches).network.params
    asym_params = run(get_config, batches, asym_critic=True).network.params
    same = jax.tree_util.tree_all(jax.tree_util.tree_map(np.array_equal, params, asym_params))
    assert same, f'{name}: asym_critic with states == observations differs from the original agent.'
    print(f'{name}: identical to the original agent after {NUM_UPDATES} updates (states == observations)')


def check_pixels(name, get_config):
    rng = np.random.default_rng(0)
    batches = [make_batch(rng, (64, 64, 9), np.uint8) for _ in range(2)]
    agent = run(get_config, batches, asym_critic=True, encoder='impala_small')
    params = agent.network.params
    critic_modules = [k for k in ['modules_critic', 'modules_value'] if k in params]
    for k in critic_modules:
        assert 'encoder' not in params[k], f'{name}: {k} has an encoder.'
    actor_modules = [k for k in params if 'actor' in k and not k.endswith('_encoder')]
    for k in actor_modules:
        # The encoder is either inside the actor or registered separately as `<actor>_encoder`.
        assert 'encoder' in params[k] or f'{k}_encoder' in params, f'{name}: {k} has no encoder.'

    sample_kwargs = dict(states=batches[0]['states'][0]) if name == 'ifql' else dict()
    action = agent.sample_actions(batches[0]['observations'][0], seed=jax.random.PRNGKey(0), **sample_kwargs)
    assert action.shape == (ACTION_DIM,)
    print(f'{name}: pixel actor + state critic OK (state-only modules: {critic_modules}, actor: {actor_modules})')


def check_critic_observations():
    """FQL with separate critic observations equal to the observations must equal the original FQL."""
    rng = np.random.default_rng(0)
    batches = [make_batch(rng, (64, 64, 9), np.uint8) for _ in range(NUM_UPDATES)]
    for batch in batches:
        batch['critic_observations'] = batch['observations']
        batch['next_critic_observations'] = batch['next_observations']

    def run_fql(**kwargs):
        config = small_config(fql_config, encoder='impala_small', **kwargs)
        agent = agents['fql'].create(
            0, batches[0]['observations'], batches[0]['actions'], config,
            ex_critic_observations=batches[0]['critic_observations'],
        )
        for batch in batches:
            agent, _ = agent.update(batch)
        return agent.network.params

    same = jax.tree_util.tree_all(
        jax.tree_util.tree_map(np.array_equal, run_fql(), run_fql(critic_observations=True))
    )
    assert same, 'fql: critic_observations == observations differs from the original agent.'
    print(f'fql: critic_observations identical to the original agent after {NUM_UPDATES} updates (pixels)')


def main():
    for name, get_config in [('fql', fql_config), ('iql', iql_config), ('ifql', ifql_config), ('sac', sac_config)]:
        check_equivalence(name, get_config)
        check_pixels(name, get_config)
    check_critic_observations()
    print('PASSED')


if __name__ == '__main__':
    main()
