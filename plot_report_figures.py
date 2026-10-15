"""Draw the figures used in the report from results/*.json -> figures/."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, 'results'); O = os.path.join(ROOT, 'figures')
os.makedirs(O, exist_ok=True)
res = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(R + '/*.json')}
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})
C = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
def ma(x, k=5):
    x = np.asarray(x); return np.convolve(np.pad(x, (k//2, k//2), mode='edge'), np.ones(k)/k, mode='valid')

def curves(name, out, title):
    r = res[name]; h = r['history']; ep = np.arange(1, len(h['train_acc'])+1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(ep, h['train_acc'], C[0], lw=1.2, label='Train'); ax[0].plot(ep, h['test_acc'], C[1], lw=1.2, label='Test')
    ax[0].scatter([r['best_epoch']], [r['best_acc']], c='k', s=18, zorder=5, label=f"Highest test {r['best_acc']:.2f}% (epoch {r['best_epoch']})")
    ax[0].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy'); ax[0].legend(fontsize=8)
    ax[1].plot(ep, h['train_loss'], C[0], lw=1.2, label='Train'); ax[1].plot(ep, h['test_loss'], C[1], lw=1.2, label='Test')
    ax[1].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Loss'); ax[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(f'{O}/{out}', dpi=200); plt.close(fig)

def compare(items, out, loss_ylim=None):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for i, (n, lab) in enumerate(items):
        h = res[n]['history']; ep = np.arange(1, len(h['train_acc'])+1); c = C[i]
        ax[0].plot(ep, ma(h['train_acc']), c, lw=1.2, label=lab)
        ax[1].plot(ep, ma(h['test_acc']), c, lw=1.2, label=lab)
        ax[2].plot(ep, ma(h['test_loss']), c, lw=1.2, label=lab)
    for a, t, y in zip(ax, ['Training accuracy', 'Testing accuracy', 'Testing loss'], ['Accuracy (%)', 'Accuracy (%)', 'Loss']):
        a.set(title=t, xlabel='Epoch', ylabel=y)
    if loss_ylim: ax[2].set_ylim(*loss_ylim)
    ax[1].legend(fontsize=7.5); fig.tight_layout(); fig.savefig(f'{O}/{out}', dpi=200); plt.close(fig)

curves('tune_a0.1_aug300_s0', 'best_curves.png', 'Best setting: EEGNet, ELU α=0.1, cosine LR, augmentation, 300 epochs')
curves('eegnet_elu_a1.0', 'baseline_curves.png', 'Baseline: EEGNet, ELU α=1.0, dropout 0.25, Adam lr 1e-2, 150 epochs')
compare([('eegnet_elu_a1.0', 'ELU (α=1.0)'), ('eegnet_relu', 'ReLU'), ('eegnet_leakyrelu', 'LeakyReLU (0.01)')], 'cmp_activation.png')
compare([('eegnet_elu_a0.1', 'α=0.1'), ('eegnet_elu_a0.5', 'α=0.5'), ('eegnet_elu_a1.0', 'α=1.0'), ('eegnet_elu_a1.5', 'α=1.5')], 'cmp_alpha.png')
compare([('eegnet_dp0.1', 'p=0.1'), ('eegnet_elu_a1.0', 'p=0.25'), ('eegnet_dp0.5', 'p=0.5')], 'cmp_dropout.png')
compare([('eegnet_elu_a1.0', 'lr 1e-2 (constant)'), ('eegnet_lr1e-3', 'lr 1e-3 (constant)'), ('eegnet_cosine', 'lr 1e-2 + cosine'), ('eegnet_true_separable', 'true separable conv')], 'cmp_lr_sep.png')
compare([('tune_a0.1_cos300_s0', 'cosine'), ('tune_a0.1_lr1e-3_300_s0', 'lr 1e-3, wd 1e-4'), ('tune_a0.1_aug300_s0', 'cosine + augmentation'), ('tune_a0.1_ls_aug300_s0', '+ label smoothing')], 'cmp_tuning.png')
compare([('eegnet_elu_a1.0', 'EEGNet (ELU)'), ('deepconvnet_elu', 'DeepConvNet ELU'), ('deepconvnet_relu', 'DeepConvNet ReLU'), ('deepconvnet_leakyrelu', 'DeepConvNet LeakyReLU')], 'cmp_deepconvnet.png', loss_ylim=(0.3, 1.2))

# ELU function
x = np.linspace(-4, 3, 400); plt.figure(figsize=(5.2, 3.1))
for i, a in enumerate([0.1, 0.5, 1.0, 1.5]):
    plt.plot(x, np.where(x > 0, x, a*(np.exp(x)-1)), C[i], lw=1.4, label=f'ELU, α={a}')
plt.plot(x, np.maximum(x, 0), 'k--', lw=1, label='ReLU')
plt.axhline(0, c='gray', lw=0.5); plt.axvline(0, c='gray', lw=0.5)
plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(fontsize=7.5); plt.ylim(-1.7, 3); plt.tight_layout(); plt.savefig(f'{O}/elu.png', dpi=200); plt.close()

# seeds
cfg = [('tune_a0.1_aug300', 'ELU α=0.1\ncosine + aug.'), ('tune_a0.1_lr1e-3_300', 'ELU α=0.1\nlr 1e-3'), ('tune_relu_lr1e-3_300', 'ReLU\nlr 1e-3')]
plt.figure(figsize=(5.4, 3.1))
for i, (k, lab) in enumerate(cfg):
    v = [res[f'{k}_s{s}']['best_acc'] for s in range(4)]
    plt.scatter(i+np.array([-0.06,-0.02,0.02,0.06]), v, c=C[i], s=28, zorder=3); plt.hlines(np.mean(v), i-0.25, i+0.25, colors='k', lw=1.5)
    plt.text(i+0.28, np.mean(v), f'mean {np.mean(v):.2f}', va='center', fontsize=8)
plt.axhline(87, c='r', ls='--', lw=1, label='87% threshold'); plt.xticks(range(3), [c[1] for c in cfg])
plt.xlim(-0.5, 2.9); plt.ylabel('Highest test accuracy (%)'); plt.legend(fontsize=8, loc='lower left'); plt.tight_layout(); plt.savefig(f'{O}/seeds.png', dpi=200); plt.close()
print('ok')
