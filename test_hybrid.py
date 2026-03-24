import torch
from hybrid_model import MultimodalModel

# Dummy inputs (same shapes as your outputs)
flow = torch.randn(2, 30, 256)
skel2d = torch.randn(2, 30, 256)
skel3d = torch.randn(2, 30, 256)

model = MultimodalModel()

output, attn = model(flow, skel2d, skel3d)

print("Final Output Shape:", output.shape)
print("Attention Shape:", attn.shape)