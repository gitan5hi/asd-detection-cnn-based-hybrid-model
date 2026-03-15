import torch
import torch.nn as nn
import torchvision.models as models


class OptimalFlowCNN(nn.Module):

    def __init__(self, output_dim=256):
        super().__init__()

        # Use pretrained ResNet18
        resnet = models.resnet18(pretrained=True)

        # Modify first layer (Optimal flow has 2 channels instead of RGB 3)
        resnet.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Final projection layer
        self.fc = nn.Linear(512, output_dim)


    def forward(self, x):
        """
        x shape:
        (batch, frames, channels, H, W)
        """

        B, T, C, H, W = x.shape

        # Merge batch + frames
        x = x.view(B*T, C, H, W)

        # Extract CNN features
        features = self.feature_extractor(x)

        # Flatten
        features = features.view(features.size(0), -1)

        # Reduce feature dimension
        features = self.fc(features)

        # Restore time dimension
        features = features.view(B, T, -1)

        return features