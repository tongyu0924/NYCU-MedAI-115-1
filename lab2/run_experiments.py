"""Run every experiment in the report with the PyTorch code, then draw comparison plots.

    python run_experiments.py            # all runs
    python run_experiments.py --plot     # only redraw comparison plots from results/*.json
"""
import os
import sys
import json
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    '-name eegnet_elu_a1.0 -activation elu -alpha 1.0',
    '-name eegnet_relu -activation relu',
    '-name eegnet_leakyrelu -activation leakyrelu',
    '-name eegnet_elu_a0.1 -activation elu -alpha 0.1',
    '-name eegnet_elu_a0.5 -activation elu -alpha 0.5',
    '-name eegnet_elu_a1.5 -activation elu -alpha 1.5',
    '-name eegnet_dp0.1 -dropout 0.1',
    '-name eegnet_dp0.5 -dropout 0.5',
    '-name eegnet_lr1e-3 -lr 0.001',
    '-name eegnet_cosine -cosine',
    '-name eegnet_true_separable -true_separable',
    '-name deepconvnet_elu -model deepconvnet -dropout 0.5 -activation elu',
    '-name deepconvnet_relu -model deepconvnet -dropout 0.5 -activation relu',
    '-name deepconvnet_leakyrelu -model deepconvnet -dropout 0.5 -activation leakyrelu',
]

GROUPS = {
    'activation': ['eegnet_elu_a1.0', 'eegnet_relu', 'eegnet_leakyrelu'],
    'elu_alpha': ['eegnet_elu_a0.1', 'eegnet_elu_a0.5', 'eegnet_elu_a1.0', 'eegnet_elu_a1.5'],
    'dropout': ['eegnet_dp0.1', 'eegnet_elu_a1.0', 'eegnet_dp0.5'],
    'lr_schedule': ['eegnet_elu_a1.0', 'eegnet_lr1e-3', 'eegnet_cosine'],
    'separable': ['eegnet_elu_a1.0', 'eegnet_true_separable'],
    'deepconvnet': ['deepconvnet_elu', 'deepconvnet_relu', 'deepconvnet_leakyrelu'],
}


def load_results():
    res = {}
    for f in glob.glob(os.path.join(ROOT, 'results', '*.json')):
        with open(f) as fh:
            res[os.path.splitext(os.path.basename(f))[0]] = json.load(fh)
    return res


def compare_plots(res):
    os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
    for g, names in GROUPS.items():
        names = [n for n in names if n in res]
        if not names:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for n in names:
            h = res[n]['history']
            ax[0].plot(h['train_acc'], '--', alpha=0.6, label=f'{n} train')
            ax[0].plot(h['test_acc'], label=f'{n} test')
            ax[1].plot(h['test_loss'], label=n)
        ax[0].set_title('Accuracy (%)'); ax[1].set_title('Test loss')
        for a in ax:
            a.set_xlabel('Epoch'); a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, 'figures', f'compare_{g}.png'), dpi=150)
        plt.close(fig)

    print(f'{"run":28s} {"best":>7s} {"epoch":>6s} {"final":>7s}')
    for n, r in sorted(res.items(), key=lambda kv: -kv[1]['best_acc']):
        print(f'{n:28s} {r["best_acc"]:7.2f} {r["best_epoch"]:6d} {r["final_acc"]:7.2f}')


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        from main import get_parser, run
        for e in EXPERIMENTS:
            run(get_parser().parse_args(e.split()))
    compare_plots(load_results())
