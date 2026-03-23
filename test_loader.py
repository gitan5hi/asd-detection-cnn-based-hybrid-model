<<<<<<< HEAD
from dataset_loader import MMASDDataset
from torch.utils.data import DataLoader

# Create Dataset
dataset = MMASDDataset()

# Create DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Try Loading ONE batch
for batch in loader:

    print("Optimal Flow shape:", batch["optimal_flow"].shape)
    print("2D Skeleton shape:", batch["skeleton2d"].shape)
    print("3D Skeleton shape:", batch["skeleton3d"].shape)
    print("Meta shape:", batch["meta"].shape)
    print("Label shape:", batch["label"].shape)

=======
from dataset_loader import MMASDDataset
from torch.utils.data import DataLoader

# Create Dataset
dataset = MMASDDataset()

# Create DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Try Loading ONE batch
for batch in loader:

    print("Optimal Flow shape:", batch["optimal_flow"].shape)
    print("2D Skeleton shape:", batch["skeleton2d"].shape)
    print("3D Skeleton shape:", batch["skeleton3d"].shape)
    print("Meta shape:", batch["meta"].shape)
    print("Label shape:", batch["label"].shape)

>>>>>>> 78b0cbccef2cbdbab00c30f83dd5e7e4cd2c51a9
    break