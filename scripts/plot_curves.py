"""Plot evaluation curves (mean and 95% bootstrap CI over seeds) and print a summary table.

Each `--run` is `label=glob`, where the glob matches run directories (one per seed) containing `eval.csv` and
`flags.json`. The x-axis is online environment steps (`step - offline_steps`), or training steps for offline runs.

Usage:
    python scripts/plot_curves.py \
        --run="Image critic=exp/fql/RLPD/img_*" --run="State critic=exp/fql/RLPD/state_*" \
        --out=exp/plots/rlpd_cube_single.png
"""

import csv
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from absl import app, flags  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('run', None, 'Series as "label=glob[,glob...]" (one directory per seed).')
flags.DEFINE_string('out', 'exp/plots/curves.png', 'Output image path (a CSV with the same name is also written).')
flags.DEFINE_string('metric', 'evaluation/success', 'Column of eval.csv to plot.')
flags.DEFINE_string('title', None, 'Plot title.')
flags.DEFINE_integer('num_last', 3, 'Number of last evaluations to average in the summary (OGBench protocol).')
flags.DEFINE_integer('num_bootstrap', 10000, 'Number of bootstrap samples for the confidence interval.')

# Categorical slots 1-4 of the validated reference palette, assigned in fixed order; each series also gets its own
# marker so that identity is never carried by color alone.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load_run(run_dir, metric):
    """Return the x values, metric values, and whether the run has online steps."""
    with open(os.path.join(run_dir, 'flags.json')) as f:
        run_flags = json.load(f)
    with open(os.path.join(run_dir, 'eval.csv')) as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r['step']) for r in rows])
    values = np.array([float(r[metric]) for r in rows])
    offline_steps, online = run_flags.get('offline_steps', 0), run_flags.get('online_steps', 0) > 0
    if not online:
        return steps, values, online
    # Online environment steps. Step 0 is the evaluation at the end of offline training, or the one after the first
    # step when there is no offline training.
    x = np.where(steps == 1, 0, steps) if offline_steps == 0 else steps - offline_steps
    return x[x >= 0], values[x >= 0], online


def format_steps(v, _):
    if v >= 1e6:
        return f'{v / 1e6:g}M'
    return f'{v / 1e3:g}K' if v else '0'


def bootstrap_ci(values, num_bootstrap, rng):
    """95% bootstrap CI of the mean over seeds (rows of `values`)."""
    idxs = rng.integers(0, len(values), (num_bootstrap, len(values)))
    means = values[idxs].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def main(_):
    rng = np.random.default_rng(0)
    series = []
    for spec in FLAGS.run:
        label, pattern = spec.split('=', 1)
        matches = [d for p in pattern.split(',') for d in glob.glob(p)]
        run_dirs = sorted(d for d in set(matches) if os.path.exists(os.path.join(d, 'eval.csv')))
        assert run_dirs, f'No runs match {pattern}'
        runs = [load_run(d, FLAGS.metric) for d in run_dirs]
        # Keep the evaluation steps that every seed has (runs may still be in progress).
        common = sorted(set.intersection(*[set(x.tolist()) for x, _, _ in runs]))
        values = np.array([[dict(zip(x.tolist(), v))[s] for s in common] for x, v, _ in runs]) * 100
        lo, hi = bootstrap_ci(values, FLAGS.num_bootstrap, rng)
        online = any(o for _, _, o in runs)
        series.append(dict(label=label, x=np.array(common), values=values, lo=lo, hi=hi, online=online))

    # Summary table.
    print(f'{"Series":24s} {"Seeds":>5s} {"Last eval":>10s} {f"Last {FLAGS.num_last} avg":>16s} {"At step":>9s}')
    for s in series:
        last, last_n = s['values'][:, -1], s['values'][:, -FLAGS.num_last :].mean(axis=1)
        print(
            f'{s["label"]:24s} {len(s["values"]):5d} {last.mean():6.1f} ± {last.std():4.1f}'
            f' {last_n.mean():10.1f} ± {last_n.std():4.1f} {s["x"][-1]:9d}'
        )

    # Plot.
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, s in enumerate(series):
        mean = s['values'].mean(axis=0)
        ax.fill_between(s['x'], s['lo'], s['hi'], color=COLORS[i], alpha=0.18, linewidth=0)
        ax.plot(s['x'], mean, color=COLORS[i], linewidth=2, marker=MARKERS[i], markersize=6, label=s['label'],
                markeredgecolor=SURFACE, markeredgewidth=1)

    # Direct labels at the line ends, pushed apart so that they do not overlap.
    ends = sorted((s['values'].mean(axis=0)[-1], i) for i, s in enumerate(series))
    label_ys, min_gap = [], 7.0
    for y, _ in ends:
        label_ys.append(max(y, label_ys[-1] + min_gap) if label_ys else y)
    for (y, i), label_y in zip(ends, label_ys):
        s = series[i]
        ax.annotate(f' {s["label"]}', (s['x'][-1], y), xytext=(s['x'][-1], label_y), color=TEXT_PRIMARY,
                    fontsize=9, va='center', annotation_clip=False)

    online = any(s['online'] for s in series)
    ax.set_xlabel('Online environment steps' if online else 'Training steps', color=TEXT_SECONDARY)
    ax.set_ylabel('Success rate (%)', color=TEXT_SECONDARY)
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_steps))
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc='lower right')
    num_seeds = sorted({len(s['values']) for s in series})
    ax.set_title(FLAGS.title or '', color=TEXT_PRIMARY, loc='left')
    if max(num_seeds) > 1:
        note = f'Mean over {"/".join(map(str, num_seeds))} seeds; shaded: 95% bootstrap CI over seeds'
    else:
        note = 'Single seed (no confidence interval)'
    fig.text(0.01, 0.01, note, color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 0.92, 1))
    os.makedirs(os.path.dirname(FLAGS.out) or '.', exist_ok=True)
    fig.savefig(FLAGS.out, dpi=200)

    # Table view of the plotted data.
    csv_path = os.path.splitext(FLAGS.out)[0] + '.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['series', 'x', 'mean', 'ci_low', 'ci_high', 'num_seeds'])
        for s in series:
            for j, x in enumerate(s['x']):
                writer.writerow([s['label'], x, s['values'][:, j].mean(), s['lo'][j], s['hi'][j], len(s['values'])])
    print(f'Saved: {FLAGS.out} and {csv_path}')


if __name__ == '__main__':
    flags.mark_flag_as_required('run')
    app.run(main)
