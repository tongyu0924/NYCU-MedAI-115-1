import os
import json
import random
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import dataloader
from models.EEGNet import EEGNet, DeepConvNet

ROOT = os.path.dirname(os.path.abspath(__file__))


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


# ----------------------------------------------------------------- plotting
def _plot_curve(values, epochs, ylabel, title, path):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), values)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_train_acc(train_acc_list, epochs, path='train_acc.png'):
    _plot_curve(train_acc_list, epochs, 'Accuracy (%)', 'Training accuracy', path)


def plot_train_loss(train_loss_list, epochs, path='train_loss.png'):
    _plot_curve(train_loss_list, epochs, 'Loss', 'Training loss', path)


def plot_test_acc(test_acc_list, epochs, path='test_acc.png'):
    _plot_curve(test_acc_list, epochs, 'Accuracy (%)', 'Testing accuracy', path)


def plot_history(hist, title, path):
    """Train/test accuracy and loss side by side."""
    ep = range(1, len(hist['train_acc']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, hist['train_acc'], label='train')
    ax[0].plot(ep, hist['test_acc'], label='test')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy (%)'); ax[0].set_title('Accuracy')
    ax[1].plot(ep, hist['train_loss'], label='train')
    ax[1].plot(ep, hist['test_loss'], label='test')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Cross-entropy loss'); ax[1].set_title('Loss')
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------- augmentation
def augment(x, args):
    """Training-time augmentation on a batch [B, 1, 2, T]."""
    if args.aug_shift > 0:
        # random circular time shift, a different shift for every trial
        B, T = x.size(0), x.size(-1)
        shifts = torch.randint(-args.aug_shift, args.aug_shift + 1, (B,), device=x.device)
        idx = (torch.arange(T, device=x.device)[None, :] - shifts[:, None]) % T
        x = torch.gather(x, -1, idx[:, None, None, :].expand_as(x))
    if args.aug_scale > 0:
        # random amplitude scaling per trial
        scale = 1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * args.aug_scale
        x = x * scale
    if args.aug_noise > 0:
        # gaussian noise relative to the signal std
        x = x + torch.randn_like(x) * args.aug_noise * x.std()
    return x


# ----------------------------------------------------------- train / test
def evaluate(model, loader, criterion, device):
    criterion = nn.CrossEntropyLoss()  # report plain CE on the test set
    model.eval()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return correct / n * 100, total_loss / n


def train(model, train_loader, test_loader, criterion, optimizer, args, device, scheduler=None):
    hist = {'train_acc': [], 'train_loss': [], 'test_acc': [], 'test_loss': []}
    best_acc, best_epoch, best_wts = 0.0, 0, None

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        correct, total_loss = 0, 0.0
        for inputs, labels in tqdm(train_loader, leave=False, desc=f'Epoch {epoch}'):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = augment(inputs, args)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        if scheduler is not None:
            scheduler.step()

        n = len(train_loader.dataset)
        train_acc, train_loss = correct / n * 100, total_loss / n
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        for k, v in zip(hist, (train_acc, train_loss, test_acc, test_loss)):
            hist[k].append(v)

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            best_wts = deepcopy(model.state_dict())  # deepcopy! state_dict() is a live reference

        print(f'Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:6.2f}% | '
              f'test loss {test_loss:.4f} acc {test_acc:6.2f}% | best {best_acc:6.2f}% (ep {best_epoch})')

    return hist, best_acc, best_epoch, best_wts


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    if args.model == 'eegnet':
        return EEGNet(activation=args.activation, alpha=args.alpha, dropout=args.dropout,
                      true_separable=args.true_separable, F1=args.F1, D=args.D, F2=args.F2)
    if args.model == 'deepconvnet':
        return DeepConvNet(activation=args.activation, alpha=args.alpha, dropout=args.dropout)
    raise ValueError(args.model)


def run(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_loader = DataLoader(BCIDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(BCIDataset(test_data, test_label), batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs) if args.cosine else None

    hist, best_acc, best_epoch, best_wts = train(model, train_loader, test_loader, criterion,
                                                 optimizer, args, device, scheduler)

    name = args.name or f'{args.model}_{args.activation}'
    for d in ('weights', 'results', 'figures'):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    torch.save(best_wts, os.path.join(ROOT, 'weights', f'{name}.pt'))

    # sanity check: reload the best weights and evaluate again
    model.load_state_dict(best_wts)
    reload_acc, _ = evaluate(model, test_loader, criterion, device)
    print(f'\n[{name}] Highest testing accuracy: {best_acc:.2f}% at epoch {best_epoch} '
          f'(reloaded weights: {reload_acc:.2f}%), last-epoch: {hist["test_acc"][-1]:.2f}%')

    with open(os.path.join(ROOT, 'results', f'{name}.json'), 'w') as f:
        json.dump({'config': vars(args), 'best_acc': best_acc, 'best_epoch': best_epoch,
                   'final_acc': hist['test_acc'][-1], 'history': hist}, f)

    fig_dir = os.path.join(ROOT, 'figures')
    plot_history(hist, name, os.path.join(fig_dir, f'{name}_curves.png'))
    plot_train_acc(hist['train_acc'], args.num_epochs, os.path.join(fig_dir, f'{name}_train_acc.png'))
    plot_train_loss(hist['train_loss'], args.num_epochs, os.path.join(fig_dir, f'{name}_train_loss.png'))
    plot_test_acc(hist['test_acc'], args.num_epochs, os.path.join(fig_dir, f'{name}_test_acc.png'))
    return best_acc


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-model', type=str, default='eegnet', choices=['eegnet', 'deepconvnet'])
    p.add_argument('-activation', type=str, default='elu', choices=['elu', 'relu', 'leakyrelu'])
    p.add_argument('-alpha', type=float, default=1.0, help='ELU alpha')
    p.add_argument('-dropout', type=float, default=0.25)
    p.add_argument('-num_epochs', type=int, default=150)
    p.add_argument('-batch_size', type=int, default=64)
    p.add_argument('-lr', type=float, default=0.01)
    p.add_argument('-weight_decay', type=float, default=0.001)
    p.add_argument('-true_separable', action='store_true',
                   help='EEGNet: depthwise (1,15) + pointwise 1x1 instead of a full (1,15) conv')
    p.add_argument('-cosine', action='store_true', help='cosine-annealing LR schedule')
    p.add_argument('-optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    p.add_argument('-label_smoothing', type=float, default=0.0)
    p.add_argument('-F1', type=int, default=16, help='EEGNet: number of temporal filters')
    p.add_argument('-D', type=int, default=2, help='EEGNet: depth multiplier (spatial filters per temporal filter)')
    p.add_argument('-F2', type=int, default=32, help='EEGNet: number of separable-conv filters')
    p.add_argument('-aug_shift', type=int, default=0, help='max random time shift (samples)')
    p.add_argument('-aug_scale', type=float, default=0.0, help='random amplitude scaling +/- ratio')
    p.add_argument('-aug_noise', type=float, default=0.0, help='gaussian noise std (relative to signal std)')
    p.add_argument('-seed', type=int, default=0)
    p.add_argument('-name', type=str, default='', help='run name used for saved files')
    return p


if __name__ == '__main__':
    run(get_parser().parse_args())
