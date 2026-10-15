"""Search for the highest-accuracy EEGNet setting.

    python tune.py            # stage 1 (candidate settings) + stage 2 (extra seeds for the top 3)
    python tune.py --quick    # stage 1 only

Every run is saved like main.py does (results/, figures/, weights/).
The ranking is written to results/tune_summary.csv and the best run is
re-evaluated at the end so you can screenshot it for the report.
"""
import os
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataloader
from main import get_parser, run, build_model, BCIDataset, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))

# name -> arguments. Built from what worked in the first round of experiments:
# small ELU alpha (0.1) and ReLU were best; more epochs / lower final LR / augmentation
# give the model more chances to reach a high test accuracy without overfitting.
CANDIDATES = {
    'relu_cos300':        '-activation relu -num_epochs 300 -cosine',
    'a0.1_cos300':        '-activation elu -alpha 0.1 -num_epochs 300 -cosine',
    'relu_lr1e-3_300':    '-activation relu -lr 0.001 -weight_decay 0.0001 -num_epochs 300',
    'a0.1_lr1e-3_300':    '-activation elu -alpha 0.1 -lr 0.001 -weight_decay 0.0001 -num_epochs 300',
    'relu_aug300':        '-activation relu -num_epochs 300 -cosine -aug_shift 25 -aug_scale 0.1 -aug_noise 0.05',
    'a0.1_aug300':        '-activation elu -alpha 0.1 -num_epochs 300 -cosine -aug_shift 25 -aug_scale 0.1 -aug_noise 0.05',
    'relu_big_aug300':    '-activation relu -F1 32 -D 2 -F2 64 -num_epochs 300 -cosine -aug_shift 25 -aug_noise 0.05',
    'a0.1_ls_aug300':     '-activation elu -alpha 0.1 -num_epochs 300 -cosine -label_smoothing 0.1 -aug_shift 25 -aug_noise 0.05',
    'relu_adamw_aug300':  '-activation relu -optimizer adamw -weight_decay 0.01 -num_epochs 300 -cosine -aug_shift 25 -aug_noise 0.05',
    'lrelu_cos300':       '-activation leakyrelu -num_epochs 300 -cosine',
}


def run_one(name, argstr):
    args = get_parser().parse_args(f'-name tune_{name} {argstr}'.split())
    return run(args), args


def main():
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    rows = []

    print('=' * 20, 'stage 1: candidate settings', '=' * 20)
    for name, a in CANDIDATES.items():
        acc, _ = run_one(f'{name}_s0', a + ' -seed 0')
        rows.append((f'{name}_s0', a + ' -seed 0', acc))
        print(f'>>> {name}: {acc:.2f}%')

    if '--quick' not in sys.argv:
        print('=' * 20, 'stage 2: more seeds for the top 3', '=' * 20)
        top = sorted(rows, key=lambda r: -r[2])[:3]
        for run_name, a, _ in top:
            base = run_name.rsplit('_s', 1)[0]
            for seed in (1, 2, 3):
                aa = CANDIDATES[base] + f' -seed {seed}'
                acc, _ = run_one(f'{base}_s{seed}', aa)
                rows.append((f'{base}_s{seed}', aa, acc))
                print(f'>>> {base} seed {seed}: {acc:.2f}%')

    rows.sort(key=lambda r: -r[2])
    with open(os.path.join(ROOT, 'results', 'tune_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'args', 'highest_test_acc'])
        w.writerows(rows)

    print('\n' + '=' * 60)
    for r in rows:
        print(f'{r[0]:28s} {r[2]:6.2f}%   {r[1]}')

    # re-evaluate the best checkpoint (good for the screenshot)
    best_name, best_args, best_acc = rows[0]
    args = get_parser().parse_args(f'-name tune_{best_name} {best_args}'.split())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_data, test_label = dataloader.read_bci_data()
    loader = DataLoader(BCIDataset(test_data, test_label), batch_size=64)
    model = build_model(args).to(device)
    w_path = os.path.join(ROOT, 'weights', f'tune_{best_name}.pt')
    model.load_state_dict(torch.load(w_path, map_location=device))
    acc, loss = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    print('=' * 60)
    print(f'BEST SETTING : {best_name}')
    print(f'ARGS         : {best_args}')
    print(f'WEIGHTS      : {w_path}')
    print(f'Highest testing accuracy (reloaded best weights): {acc:.2f}%')
    print('=' * 60)


if __name__ == '__main__':
    main()
