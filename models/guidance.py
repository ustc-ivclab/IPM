import torch.nn as nn

class GuidanceBlock(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels, guid_in_chans, stride=2):
        super(GuidanceBlock, self).__init__()
        self.intermediate_extraction = nn.Sequential(
            nn.Conv2d(guid_in_chans, in_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
        self.conv_with_intermediation = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, input, guidance):
        if guidance is None:
            return input
        else:
            intermediate_features = self.intermediate_extraction(guidance)
            input *= intermediate_features
            return self.conv_with_intermediation(input) + input

