import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['msanet_pos']


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for MSANet
##########
class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 scale,
                 pos_type=''):
        super(ContextBlock, self).__init__()

        self.inplanes = inplanes
        self.ratio = ratio
        self.scale = scale
        self.planes = int(inplanes * ratio)

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

        self.pos_type = pos_type
        if pos_type == 'PAAS':
            self.register_parameter(name='weighted_matrix', param=nn.Parameter(torch.Tensor(self.inplanes, 1)))
            nn.init.normal_(self.weighted_matrix, 0, 0.01)
        elif pos_type == 'CPE':
            self.proj = nn.Conv2d(1, 1, (5, 1), 1, (2, 0))
        elif pos_type == 'LAPE':
            self.register_parameter(name='weighted_matrix', param=nn.Parameter(torch.Tensor(self.inplanes, 1)))
            nn.init.normal_(self.weighted_matrix, 0, 0.01)
        else:
            raise TypeError('Unsupported position embedding type.')

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        if self.pos_type == 'PAAS':
            context = context / self.scale
            context = context.mul(torch.sigmoid(self.weighted_matrix))
        elif self.pos_type == 'CPE':
            context = context / self.scale
            context = context + self.proj(context)
        elif self.pos_type == 'LAPE':
            context = context / self.scale
            context = context + self.weighted_matrix
        else:
            raise TypeError('Unsupported position embedding type.')
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


class MSABlock_1(nn.Module):
    """Multi-scale aggregation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        pos_type='',
        bottleneck_reduction=4,
        **kwargs
    ):
        super(MSABlock_1, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = LightConv3x3(mid_channels, mid_channels)
        self.conv2c = LightConv3x3(mid_channels, mid_channels)
        self.gate = ContextBlock(mid_channels, 1/16, 3., pos_type=pos_type)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

        self.conv1x1_1 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)

        # bottom-up
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x2a)
        x2c = self.conv2c(x2b)

        # top-down
        x2b_td = self.conv1x1_1(x2c) + self.conv1x1_2(x2b)
        x2a_td = self.conv1x1_3(x2b_td) + self.conv1x1_4(x2a)

        x2 = self.gate(x2a_td) + self.gate(x2b_td) + self.gate(x2c)
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class MSABlock_2(nn.Module):
    """Multi-scale aggregation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        pos_type='',
        bottleneck_reduction=4,
        **kwargs
    ):
        super(MSABlock_2, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = LightConv3x3(mid_channels, mid_channels)
        self.conv2c = LightConv3x3(mid_channels, mid_channels)
        self.conv2d = LightConv3x3(mid_channels, mid_channels)
        self.gate = ContextBlock(mid_channels, 1/16, 3., pos_type=pos_type)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

        self.conv1x1_1 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_5 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_6 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)

        # bottom-up
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x2a)
        x2c = self.conv2c(x2b)
        x2d = self.conv2d(x2c)

        # top-down
        x2c_td = self.conv1x1_1(x2d) + self.conv1x1_2(x2c)
        x2b_td = self.conv1x1_3(x2c_td) + self.conv1x1_4(x2b)
        x2a_td = self.conv1x1_5(x2b_td) + self.conv1x1_6(x2a)

        x2 = self.gate(x2a_td) + self.gate(x2b_td) + self.gate(x2c_td) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class MSABlock_3(nn.Module):
    """Multi-scale aggregation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        pos_type='',
        **kwargs
    ):
        super(MSABlock_3, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = LightConv3x3(mid_channels, mid_channels)
        self.conv2c = LightConv3x3(mid_channels, mid_channels)
        self.conv2d = LightConv3x3(mid_channels, mid_channels)
        self.conv2e = LightConv3x3(mid_channels, mid_channels)
        self.gate = ContextBlock(mid_channels, 1/16, 3., pos_type=pos_type)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

        self.conv1x1_1 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_5 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_6 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_7 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1x1_8 = nn.Conv2d(
            mid_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)

        # bottom-up
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x2a)
        x2c = self.conv2c(x2b)
        x2d = self.conv2d(x2c)
        x2e = self.conv2e(x2d)

        # top-down
        x2d_td = self.conv1x1_1(x2e) + self.conv1x1_2(x2d)
        x2c_td = self.conv1x1_3(x2d_td) + self.conv1x1_4(x2c)
        x2b_td = self.conv1x1_5(x2c_td) + self.conv1x1_6(x2b)
        x2a_td = self.conv1x1_7(x2b_td) + self.conv1x1_8(x2a)

        x2 = self.gate(x2a_td) + self.gate(x2b_td) + self.gate(x2c_td) + self.gate(x2d_td) + self.gate(x2e)
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


##########
# Network architecture
##########
class MSANet(nn.Module):
    """Multi-Scale Aggregation Network.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        conv1_IN=False,
        IN=[],
        pos_type='',
        input_size=(192, 96),
        feature_scales=(4, 8, 16, 16),
        pooling_type='conv',
        **kwargs
    ):
        super(MSANet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1

        self.feature_dim = feature_dim
        self.input_size = input_size
        self.feature_scales = feature_scales

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN[0],
            pos_type=pos_type
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
            IN=IN[1],
            pos_type=pos_type
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
            IN=IN[2],
            pos_type=pos_type
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        
        if pooling_type == 'avg':
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            kernel_size = (input_size[0] // self.feature_scales[-1], input_size[1] // self.feature_scales[-1])
            self.global_avgpool = nn.Conv2d(channels[3], channels[3], kernel_size, groups=channels[3])

        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )

        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=[],
        pos_type=''
    ):
        layers = []

        layers.append(block[0](in_channels, out_channels, IN=IN[0], pos_type=pos_type))
        for i in range(1, layer):
            layers.append(block[i](out_channels, out_channels, IN=IN[i], pos_type=pos_type))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.PReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        # stage 1
        x = self.conv2(x)

        # stage 2
        x = self.conv3(x)

        # stage 3
        x = self.conv4(x)

        x = self.conv5(x)
        return x

    def forward(self, x):
        if self.training:
            x = self.featuremaps(x)

            f = self.global_avgpool(x)
            f = f.view(f.size(0), -1)
            f = self.fc(f)
            y = self.classifier(f)

            return f, y

        else:
            x = self.featuremaps(x)

            f = self.global_avgpool(x)
            f = f.view(f.size(0), -1)
            f = self.fc(f)

            return f


##########
# Instantiation
##########
def msanet_pos(
    num_classes=1000, **kwargs
):
    pos_type = 'PAAS'

    model = MSANet(
        num_classes,
        blocks=[[MSABlock_1, MSABlock_2], [MSABlock_2, MSABlock_2], [MSABlock_2, MSABlock_3]],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        feature_dim=512,
        conv1_IN=True,
        IN=[[True, True], [False, True], [True, False]],
        pos_type=pos_type,
        **kwargs
    )

    return model
