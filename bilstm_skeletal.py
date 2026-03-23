<<<<<<< HEAD
import torch
import torch.nn as nn

class SkeletonBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(SkeletonBiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, 256)

    def forward(self, x):

        # x shape → (B, T, J, C)
        B, T, J, C = x.shape

        # Flatten joints
        x = x.view(B, T, J * C)

        # LSTM
        out, _ = self.lstm(x)

        # Output projection
        out = self.fc(out)

        return out

skeleton2d_model = SkeletonBiLSTM(input_size=75)   # 25 × 3
skeleton3d_model = SkeletonBiLSTM(input_size=72)   # 24 × 3

if __name__ == "__main__":

    B = 2
    T = 30

    skeleton2d = torch.randn(B, T, 25, 3)
    skeleton3d = torch.randn(B, T, 24, 3)

    model2d = SkeletonBiLSTM(75)
    model3d = SkeletonBiLSTM(72)

    out2d = model2d(skeleton2d)
    out3d = model3d(skeleton3d)

    print("2D BiLSTM Output:", out2d.shape)
=======
import torch
import torch.nn as nn

class SkeletonBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(SkeletonBiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, 256)

    def forward(self, x):

        # x shape → (B, T, J, C)
        B, T, J, C = x.shape

        # Flatten joints
        x = x.view(B, T, J * C)

        # LSTM
        out, _ = self.lstm(x)

        # Output projection
        out = self.fc(out)

        return out

skeleton2d_model = SkeletonBiLSTM(input_size=75)   # 25 × 3
skeleton3d_model = SkeletonBiLSTM(input_size=72)   # 24 × 3

if __name__ == "__main__":

    B = 2
    T = 30

    skeleton2d = torch.randn(B, T, 25, 3)
    skeleton3d = torch.randn(B, T, 24, 3)

    model2d = SkeletonBiLSTM(75)
    model3d = SkeletonBiLSTM(72)

    out2d = model2d(skeleton2d)
    out3d = model3d(skeleton3d)

    print("2D BiLSTM Output:", out2d.shape)
>>>>>>> 78b0cbccef2cbdbab00c30f83dd5e7e4cd2c51a9
    print("3D BiLSTM Output:", out3d.shape)