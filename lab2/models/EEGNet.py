import torch.nn as nn


def get_activation(name='elu', alpha=1.0):
    """Return an activation module by name."""
    name = name.lower()
    if name == 'elu':
        return nn.ELU(alpha=alpha)
    if name == 'relu':
        return nn.ReLU()
    if name in ('leakyrelu', 'leaky_relu', 'lrelu'):
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f'Unknown activation: {name}')


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018) for input [B, 1, 2, 750].

    firstconv      : temporal Conv2d(1->F1, (1,51))      - learns band-pass filters
    depthwiseConv  : Conv2d(F1->F1*D, (C,1), groups=F1)  - spatial filter per temporal filter
    separableConv  : Conv2d(F1*D->F2, (1,15))            - temporal summary of each feature map
    classify       : Linear(F2 * 23 -> 2)
    """

    def __init__(self, activation='elu', alpha=1.0, dropout=0.25,
                 num_classes=2, channels=2, samples=750, F1=16, D=2, F2=32,
                 true_separable=False):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(channels, 1), stride=(1, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(activation, alpha),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout),
        )
        if true_separable:
            # depthwise (1,15) conv per feature map + pointwise 1x1 conv to mix maps
            sep = [nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 15), padding=(0, 7), groups=F1 * D, bias=False),
                   nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)]
        else:
            # as in the lab spec: one full (1,15) conv over all 32 maps
            sep = [nn.Conv2d(F1 * D, F2, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)]
        self.separableConv = nn.Sequential(
            *sep,
            nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(activation, alpha),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout),
        )
        # 750 -> /4 -> 187 -> /8 -> 23
        feat_len = (samples // 4) // 8
        self.classify = nn.Sequential(
            nn.Linear(in_features=F2 * feat_len, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)  # logits (CrossEntropyLoss applies log-softmax)


class DeepConvNet(nn.Module):
    """DeepConvNet (Schirrmeister et al., 2017), C=2, T=750, N=2. Max-norm omitted."""

    def __init__(self, activation='elu', alpha=1.0, dropout=0.5,
                 num_classes=2, channels=2, samples=750):
        super(DeepConvNet, self).__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(1, 5)),
                nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
                get_activation(activation, alpha),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=dropout),
            )

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(channels, 1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            get_activation(activation, alpha),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout),
        )
        self.conv1 = block(25, 50)
        self.conv2 = block(50, 100)
        self.conv3 = block(100, 200)

        # 750 -> 746 -> 373 -> 369 -> 184 -> 180 -> 90 -> 86 -> 43
        t = samples
        for _ in range(4):
            t = (t - 4) // 2
        self.classify = nn.Linear(200 * t, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)
