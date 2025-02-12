# Channel Attention layer
import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    

class SELayerModulation(nn.Module):
    """SElayer with QP/TID Modulation

    Args:
        nn (_type_): _description_
    """
    def __init__(self, channel, reduction=16):
        super(SELayerModulation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
        )
        self.fc_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction + 1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, qp):
        """The input qp should be normalized, 0.5 + qp/51"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_1(y)
        y = torch.cat([y, qp], dim=1)
        y = self.fc_2(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class ECALayerModualtion(nn.Module):
    """ECA Layer with QP/TID Modulation
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayerModualtion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
if __name__ == "__main__":
    import torch
    torch.autograd.set_detect_anomaly(True)
    net = SELayerModulation(channel=48, reduction=2).cuda()
    input_batch = torch.ones(2, 48, 128, 128).cuda()
    qp = torch.randn(2, 1).cuda()
    out = net(input_batch, qp)
    print(out.shape)