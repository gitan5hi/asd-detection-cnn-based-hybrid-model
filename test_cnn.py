<<<<<<< HEAD
import torch
from cnn_optimalflow import OptimalFlowCNN
from dataset_loader import MMASDDataset
from torch.utils.data import DataLoader

dataset = MMASDDataset()
loader = DataLoader(dataset, batch_size=2)

model = OptimalFlowCNN()

for batch in loader:

    optimal_flow = batch["optimal_flow"]

    output = model(optimal_flow)

    print("CNN Output Shape:", output.shape)

=======
import torch
from cnn_optimalflow import OptimalFlowCNN
from dataset_loader import MMASDDataset
from torch.utils.data import DataLoader

dataset = MMASDDataset()
loader = DataLoader(dataset, batch_size=2)

model = OptimalFlowCNN()

for batch in loader:

    optimal_flow = batch["optimal_flow"]

    output = model(optimal_flow)

    print("CNN Output Shape:", output.shape)

>>>>>>> 78b0cbccef2cbdbab00c30f83dd5e7e4cd2c51a9
    break