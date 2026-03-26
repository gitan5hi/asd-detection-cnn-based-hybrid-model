import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_loader import MMASDDataset
from hybrid_model import MultimodalModel
from cnn_optimalflow import OptimalFlowCNN
from bilstm_skeletal import SkeletonBiLSTM

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
BATCH_SIZE = 2
EPOCHS = 3
LR = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# DATASET + SPLIT
# ==============================
dataset = MMASDDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ==============================
# MODELS
# ==============================
cnn = OptimalFlowCNN().to(device)
bilstm2d = SkeletonBiLSTM(input_size=75).to(device)
bilstm3d = SkeletonBiLSTM(input_size=72).to(device)
model = MultimodalModel().to(device)

# ==============================
# LOSS + OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(cnn.parameters()) +
    list(bilstm2d.parameters()) +
    list(bilstm3d.parameters()) +
    list(model.parameters()),
    lr=LR
)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1} =====")

    cnn.train()
    bilstm2d.train()
    bilstm3d.train()
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for i, batch in enumerate(train_loader):

        flow = batch["optimal_flow"].to(device)
        skel2d = batch["skeleton2d"].to(device)
        skel3d = batch["skeleton3d"].to(device)
        label = batch["label"].to(device)

        # ===== FORWARD =====
        flow_feat = cnn(flow)
        skel2d_feat = bilstm2d(skel2d)
        skel3d_feat = bilstm3d(skel3d)

        outputs, _ = model(flow_feat, skel2d_feat, skel3d_feat)

        # ===== LOSS =====
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

        if i % 50 == 0:
            print(f"Batch {i} running...")

    # ===== TRAIN METRICS =====
    train_accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    print(f"\nTrain Loss: {avg_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    # ==============================
    # TEST / EVALUATION
    # ==============================
    cnn.eval()
    bilstm2d.eval()
    bilstm3d.eval()
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:

            flow = batch["optimal_flow"].to(device)
            skel2d = batch["skeleton2d"].to(device)
            skel3d = batch["skeleton3d"].to(device)
            label = batch["label"].to(device)

            flow_feat = cnn(flow)
            skel2d_feat = bilstm2d(skel2d)
            skel3d_feat = bilstm3d(skel3d)

            outputs, _ = model(flow_feat, skel2d_feat, skel3d_feat)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ===== METRICS =====
    test_accuracy = 100 * (sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels))
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # ===== ROC CURVE =====
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()