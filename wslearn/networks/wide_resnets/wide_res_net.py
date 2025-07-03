
# This implementation is modified from https://github.com/microsoft/Semi-supervised-learning
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    """
    A ResNet basic block
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride,
                 drop_rate=0.0,
                 momentum=0.001,
                 activate_before_residual=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes,
                               out_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes,
                               out_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)

        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(in_planes,
                                           out_planes,
                                           kernel_size=1,
                                           stride=stride,
                                           padding=0,
                                           bias=False)
        else:
            self.conv_shortcut = None

        self.activate_before_residual = activate_before_residual

    def forward(self, x):

        if not self.equal_in_out and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        if self.equal_in_out:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if self.equal_in_out:
            return torch.add(x, out)
        return torch.add(self.conv_shortcut(x), out)

class _NetworkBlock(nn.Module):
    """
    A group of ResNet Basic blocks
    """
    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 stride,
                 drop_rate=0.0,
                 momentum=0.001,
                 activate_before_residual=False):
        super().__init__()
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                _BasicBlock(in_planes=in_planes if i == 0 else out_planes,
                            out_planes=out_planes,
                            stride=stride if i == 0 else 1,
                            drop_rate=drop_rate,
                            momentum=momentum,
                            activate_before_residual=activate_before_residual)
                            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class _WideResNet(nn.Module):

    def __init__(
        self,
        num_classes,
        in_channels,
        depth,
        widen_factor,
        first_stride=1,
        drop_rate=0.0,
        momentum=0.001,
    ):
        super().__init__()
        channels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(in_channels,
                               channels[0],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.block1 = _NetworkBlock(nb_layers=n,
                                    in_planes=channels[0],
                                    out_planes=channels[1],
                                    stride=first_stride,
                                    drop_rate=drop_rate,
                                    momentum=momentum,
                                    activate_before_residual=True)
        self.block2 = _NetworkBlock(nb_layers=n,
                                    in_planes=channels[1],
                                    out_planes=channels[2],
                                    stride=2,
                                    drop_rate=drop_rate,
                                    momentum=momentum,
                                    activate_before_residual=False)
        self.block3 = _NetworkBlock(nb_layers=n,
                                    in_planes=channels[2],
                                    out_planes=channels[3],
                                    stride=2,
                                    drop_rate=drop_rate,
                                    momentum=momentum,
                                    activate_before_residual=False)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=momentum, eps=0.001)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.classifier = nn.Linear(channels[3], num_classes)

        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.classifier(out)

class WideResNet(nn.Module):
    """ A WideResNet
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 depth,
                 widen_factor,
                 first_stride=1,
                 drop_rate=0.0,
                 momentum=0.001):
        """ Initialise a WideResNet

        Args:
            num_classes: The number of classes for the output layer
            in_channels: The number of channels for inputs
            depth: The number of layers. Must satisfy (depth - 4) // 6 == 0
            widen_factor: The widening factor for each successive layer
            first_stride: The stride for the first NetworkBlock
            drop_rate: The droprate for dropout layers
            momentum: The momentum for BatchNorm layers
        """

        super().__init__()
        self.model = _WideResNet(num_classes=num_classes,
                                 in_channels=in_channels,
                                 depth=depth,
                                 widen_factor=widen_factor,
                                 first_stride=first_stride,
                                 drop_rate=drop_rate,
                                 momentum=momentum)

    def forward(self, x):
        return self.model(x)


class WideResNet_10_1(WideResNet):
    """ A WideResNet with depth 10 and widen factor 1"""

    def __init__(self,
                 num_classes,
                 in_channels,
                 first_stride=1,
                 drop_rate=0.0):
        """ Initialise a WideResNet with depth 10 and widen factor 1

        Args:
            num_classes: The number of classes for the output layer
            in_channels: The number of channels for inputs
            first_stride: The stride for the first NetworkBlock
            drop_rate: The droprate for dropout layers
        """
        super().__init__(num_classes,
                         in_channels,
                         depth=10,
                         widen_factor=1,
                         first_stride=first_stride,
                         drop_rate=drop_rate)


class WideResNet_28_2(WideResNet):
    """ A WideResNet with depth 28 and widen factor 2"""

    def __init__(self,
                 num_classes,
                 in_channels,
                 first_stride=1,
                 drop_rate=0.0):
        """ Initialise a WideResNet with depth 10 and widen factor 1

        Args:
            num_classes: The number of classes for the output layer
            in_channels: The number of channels for inputs
            first_stride: The stride for the first NetworkBlock
            drop_rate: The droprate for dropout layers
        """
        super().__init__(num_classes,
                         in_channels,
                         depth=28,
                         widen_factor=2,
                         first_stride=first_stride,
                         drop_rate=drop_rate)


class WideResNet_28_8(WideResNet):
    """ A WideResNet with depth 28 and widen factor 8"""

    def __init__(self,
                 num_classes,
                 in_channels,
                 first_stride=1,
                 drop_rate=0.0):
        super().__init__(num_classes,
                         in_channels,
                         depth=28,
                         widen_factor=8,
                         first_stride=first_stride,
                         drop_rate=drop_rate)

