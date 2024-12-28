from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        '''
        Args:
            in_channels: 입력 채널 차원
            out_channels: 출력 채널 차원
            stride: convolution layer의 stride
            downsample: skip connection에 의한 차원 불일치 문제 해결을 위한 변환 레이어
            act_layer: activation layer
            norm_layer: normalization layer
        '''
        super(BasicBlock, self).__init__()
        
        # 1st Conv Block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)

        # 2nd Conv Block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels * self.expansion)

        # downsample layer & activation layer
        self.downsample = downsample
        self.act = act_layer(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # 1st conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # 2nd conv block
        x = self.conv2(x)
        x = self.bn2(x)

        # skip connection & activation
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        
        '''
        Args:
            in_channels: 입력 채널 차원
            out_channels: 출력 채널 차원
            stride: convolution layer의 stride
            downsample: skip connection에 의한 차원 불일치 문제 해결을 위한 변환 레이어
            act_layer: activation layer
            norm_layer: normalization layer
        '''
        super(Bottleneck, self).__init__()

        # 1st Conv block (1x1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)

        # 2nd Conv Block (3x3)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)

        # 3rd Conv Block (1x1)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(out_channels * self.expansion)

        # downsample layer & activation layer
        self.downsample = downsample
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # 1st conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # 2nd conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # 3rd conv block
        x = self.conv3(x)
        x = self.bn3(x)

        # skip connection & activation
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)

        return x

def down_avg(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    norm_layer: Type[nn.Module] = nn.BatchNorm2d
) -> nn.Module:

    if stride == 1:
        pool = nn.Identity()
    else:
        pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=kernel_size//2, ceil_mode=True, count_include_pad=False
        )
    
    output = nn.Sequential(
        pool, nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), norm_layer(out_channels)
    )

    return output

def down_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    norm_layer: Type[nn.Module] = nn.BatchNorm2d
) -> nn.Module:
    raise NotImplementedError

class ResNet(nn.Module):
    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: Tuple[int, ...],
                 num_classes: int = 1000,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
        
def resnet18():
    raise NotImplementedError

def resnet34():
    raise NotImplementedError

def resnet50():
    raise NotImplementedError

def resnet101():
    raise NotImplementedError

def resnet152():
    raise NotImplementedError

if __name__ == "__main__":
    net = resnet18()
