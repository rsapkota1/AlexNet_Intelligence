
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AlexNet3D_Dropout_White(nn.Module):
    def __init__(self, num_classes = 100):
        super(AlexNet3D_Dropout_White, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU( inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(3456, num_classes),
                                        nn.ReLU( inplace=True),
                                       )
        #3456
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #return [x, x]
        return x