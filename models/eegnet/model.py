import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_classes: int, electrode_channels: int = 64, sample_length: int = 128,
                 dropout_rate: float = 0.5, kernel_length: int = 32, f1: int = 8,
                 d: int = 2, f2: int = 16) -> None:
        super().__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=f1,
            kernel_size=(1, kernel_length),
            padding='same',
            bias=False
        )
        self.block1_bn1 = nn.BatchNorm2d(f1, momentum=0.01, eps=1e-3)

        self.block1_conv2 = nn.Conv2d(
            in_channels=f1,
            out_channels=f1 * d,
            kernel_size=(electrode_channels, 1),
            groups=f1,
            bias=False
        )
        self.block1_bn2 = nn.BatchNorm2d(f1 * d, momentum=0.01, eps=1e-3)
        self.block1_elu = nn.ELU()
        self.block1_pool = nn.AvgPool2d((1, 4))
        self.block1_dropout = nn.Dropout(p=dropout_rate)

        # Block 2
        self.block2_conv1 = nn.Conv2d(
            in_channels=f1 * d,
            out_channels=f1 * d,
            kernel_size=(1, 16),
            padding='same',
            groups=f1 * d,
            bias=False
        )
        self.block2_conv2 = nn.Conv2d(
            in_channels=f1 * d,
            out_channels=f2,
            kernel_size=(1, 1),
            bias=False
        )
        self.block2_bn = nn.BatchNorm2d(f2, momentum=0.01, eps=1e-3)
        self.block2_elu = nn.ELU()
        self.block2_pool = nn.AvgPool2d((1, 8))
        self.block2_dropout = nn.Dropout(p=dropout_rate)

        # Classification
        self.flatten = nn.Flatten()

        # Calculate resulting temporal dimension
        feature_dim = f2 * (sample_length // 32)

        self.dense = nn.Linear(
            in_features=feature_dim,
            out_features=n_classes
        )

    def apply_max_norm(self, max_norm_value=1.0):
        """
        Industry-standard practice for EEGNet: 
        Apply a max-norm constraint to the weights of the fully connected layer 
        and the pointwise convolution to avoid overfitting on noisy data.
        """
        with torch.no_grad():
            for module in [self.block2_conv2, self.dense]:
                if hasattr(module, 'weight'):
                    norm = module.weight.norm(2, dim=0, keepdim=True)
                    desired = torch.clamp(norm, 0, max_norm_value)
                    module.weight *= (desired / (norm + 1e-8))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        kwargs allows compatibility with Trainer class standard
        """
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn(x)
        x = self.block2_elu(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        # Classification
        x = self.flatten(x)
        x = self.dense(x)

        return x
