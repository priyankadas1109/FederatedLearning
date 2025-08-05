#FederatedLearning.py
from PIL import Image
import flwr as fl
from flwr.server.strategy import FedOpt
from flwr.common import ndarrays_to_parameters
from typing import Dict, List, Tuple
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from collections import Counter
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio


# Load dataset (modify the path if needed)
metadata_path = "/Users/priyankadas/Downloads/HAM10000/HAM10000_metadata.csv" 
image_folder = "/Users/priyankadas/Downloads/HAM10000/images" 

# Load metadata CSV
df = pd.read_csv(metadata_path)

# Display basic info
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print(df.head())

### 2️ **Display Sample Images (Only 6 Images)**
def display_sample_images(image_folder, metadata_df, num_images=6):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns
    sample_images = metadata_df.sample(n=num_images, random_state=42)  # Randomly pick images

    for idx, (ax, row) in enumerate(zip(axes.flatten(), sample_images.iterrows())):
        image_path = os.path.join(image_folder, row[1]['image_id'] + ".jpg")
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_title(f"Class: {row[1]['dx']}")
        ax.axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=14)
    plt.tight_layout()
    plt.show()

display_sample_images(image_folder, df, num_images=6)

### 3️ **(Optional) Correlation Matrix - Only for Structured Data**
if df.select_dtypes(include=[np.number]).shape[1] > 1:  # Only if numerical features exist
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.show()
else:
    print("Skipping correlation matrix (no numerical features).")


# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define CNN Model
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Removed Softmax (CrossEntropyLoss expects raw logits)
        return x

# Load HAM10000 Dataset
DATASET_PATH = "/Users/priyankadas/Downloads/HAM10000"

# Load Metadata CSV (which contains labels)
metadata_path = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
metadata = pd.read_csv(metadata_path)

# Dictionary mapping image_id to disease label
label_mapping = {disease: idx for idx, disease in enumerate(metadata['dx'].unique())}
metadata['label'] = metadata['dx'].map(label_mapping)

# Custom PyTorch Dataset
class HAM10000Dataset(Dataset):
    def __init__(self, metadata, img_dir, transform=None):
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx]['image_id'] + ".jpg"
        label = self.metadata.iloc[idx]['label']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Fixed RGB normalization
])

# Load the dataset
dataset = HAM10000Dataset(metadata, os.path.join(DATASET_PATH, "images"), transform=transform)

# Split into federated clients (5 clients)
NUM_CLIENTS = 5
split_sizes = [len(dataset) // NUM_CLIENTS] * (NUM_CLIENTS - 1)
split_sizes.append(len(dataset) - sum(split_sizes))  # Ensure all data is used

# Split into train+clients and test
test_size = int(0.1 * len(dataset))
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Then split train_dataset into federated clients
split_sizes = [len(train_dataset) // NUM_CLIENTS] * (NUM_CLIENTS - 1)
split_sizes.append(len(train_dataset) - sum(split_sizes))  # Use full train set

# Initial raw splits
federated_raw = random_split(train_dataset, split_sizes)

# Upsample each client dataset
def upsample_dataset(dataset):
    from collections import Counter
    import random
    targets = [dataset[i][1] for i in range(len(dataset))]
    label_to_indices = {label: [] for label in set(targets)}
    for idx, label in enumerate(targets):
        label_to_indices[label].append(idx)
    max_count = max(len(idxs) for idxs in label_to_indices.values())
    upsampled_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) < max_count:
            oversampled = random.choices(indices, k=max_count - len(indices))
            upsampled_indices.extend(indices + oversampled)
        else:
            upsampled_indices.extend(indices)
    random.shuffle(upsampled_indices)
    return Subset(dataset, upsampled_indices)

def split_train_val(dataset, val_ratio=0.1):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

federated_datasets = []
federated_valsets = []

for ds in federated_raw:
    upsampled = upsample_dataset(ds)
    train_split, val_split = split_train_val(upsampled)
    federated_datasets.append(train_split)
    federated_valsets.append(val_split)

#federated_datasets = [upsample_dataset(client_ds) for client_ds in federated_raw]

# Optional: Print Class Distribution per Client
def print_class_distribution(dataset, client_id):
    from collections import Counter
    loader = DataLoader(dataset, batch_size=32)
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    counter = Counter(all_labels)
    print(f"Client {client_id} class distribution: {dict(counter)}")

for i, client_data in enumerate(federated_datasets):
    print_class_distribution(client_data, i)

#federated_datasets = random_split(dataset, split_sizes)

# Define Flower Client
class SkinCancerClient(fl.client.NumPyClient):
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = SkinCancerCNN().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param).to(DEVICE)  # Move to GPU

    def fit(self, parameters, config):
        #print("\n [DEBUG] Before training - Client model parameters:", self.get_parameters(config)[0][:5])

        self.set_parameters(parameters)
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(3):  
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)  # Move to GPU
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        #print(f"[DEBUG] After training - Client avg loss: {avg_loss}")
        #print("[DEBUG] After training - Updated model parameters:", self.get_parameters(config)[0][:5])

        return self.get_parameters(config), len(self.train_dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        loss_total = 0
        self.model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_loss = loss_total / total
        return avg_loss, total, {"accuracy": accuracy, "loss": avg_loss}



# Create client function (FIXED)
def client_fn(cid):
    return SkinCancerClient(federated_datasets[int(cid)], federated_valsets[int(cid)])

# Define the function to aggregate accuracy across clients
def weighted_average(metrics):
    """Aggregate accuracy across clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Store history globally
loss_history = []
accuracy_history = []

from typing import Dict, List, Tuple
import flwr as fl

class CustomFedOpt(FedOpt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_client_train_losses: Dict[int, List[float]] = {}
        self.round_client_val_losses: Dict[int, List[float]] = {}
        self.round_client_accuracies: Dict[int, List[float]] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
        # Collect loss if returned
        train_losses = []
        for client, fit_res in results:
            if fit_res.metrics and "loss" in fit_res.metrics:
                train_losses.append(fit_res.metrics["loss"])
        self.round_client_train_losses[server_round] = train_losses
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        val_losses = []
        accuracies = []
        for client, eval_res in results:
            if eval_res.metrics:
                if "loss" in eval_res.metrics:
                    val_losses.append(eval_res.metrics["loss"])
                if "accuracy" in eval_res.metrics:
                    accuracies.append(eval_res.metrics["accuracy"])

        self.round_client_val_losses[server_round] = val_losses
        self.round_client_accuracies[server_round] = accuracies
        return super().aggregate_evaluate(server_round, results, failures)

# Initialize the global model to extract initial weights
initial_model = SkinCancerCNN().to(DEVICE)
initial_weights = [val.cpu().detach().numpy() for val in initial_model.parameters()]
initial_parameters = ndarrays_to_parameters(initial_weights)


strategy = CustomFedOpt(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=initial_parameters,
    evaluate_metrics_aggregation_fn=weighted_average,

    # FedOpt-specific
    eta=0.01,     # server-side learning rate
    eta_l=0.001,  # client-side (for tracking)
    beta_1=0.9,
    beta_2=0.999,
    tau=1e-9
)



# Start the Flower simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy
)

# Evaluate one of the trained clients after FL simulation
print("\n Evaluating federated model on held-out test set...")

# Use a trained client model (all are synced post-simulation)
client_model = SkinCancerClient(train_dataset=test_dataset, val_dataset=test_dataset)  # test_dataset already defined
model = client_model.model  # This is the trained model

# Prepare test loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Plots
#1. Class Distribution

plt.figure(figsize=(8, 4))
sns.countplot(x='dx', data=metadata, order=metadata['dx'].value_counts().index, palette='viridis')
plt.title("Class Distribution in HAM10000", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Image Count")
plt.xlabel("Skin Condition")
plt.tight_layout()
plt.show()

# 2. Image Dimensions Distribution
img_sizes = []
for img_id in metadata['image_id'].values[:500]:  # limit for speed
    path = os.path.join(image_folder, img_id + ".jpg")
    if os.path.exists(path):
        with Image.open(path) as img:
            img_sizes.append(img.size)

widths, heights = zip(*img_sizes)
plt.figure(figsize=(6, 4))
sns.histplot(widths, bins=20, color='teal', label='Width', kde=True)
sns.histplot(heights, bins=20, color='orange', label='Height', kde=True)
plt.title("Distribution of Image Sizes")
plt.legend()
plt.tight_layout()
plt.show()

# 3. Model Evaluation Visualizations

# Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# After prediction on test set
y_true = []  # true labels
y_pred = []  # predicted labels

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='YlGnBu', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ROC Curve (One-vs-Rest for Multi-class)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay

# Assuming 7 classes
y_true_bin = label_binarize(y_true, classes=list(range(7)))
y_score = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE)).cpu()
        y_score.extend(outputs.numpy())

y_score = np.array(y_score)

# Plot ROC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(7):
    plt.plot(fpr[i], tpr[i], label=f'{list(label_mapping.keys())[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curve")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot dynamic surface graphs using tracked loss and accuracy

# Plotting at the end
from plotly import graph_objects as go
import numpy as np

# Extract round and client accuracy/loss
rounds = sorted(strategy.round_client_train_losses.keys())
# Build matrices (Rounds x Clients)
train_loss_matrix = [strategy.round_client_train_losses[r] for r in rounds]
val_loss_matrix   = [strategy.round_client_val_losses[r] for r in rounds]
acc_matrix        = [strategy.round_client_accuracies[r] for r in rounds]

# Convert to NumPy arrays
train_loss_matrix = np.array(train_loss_matrix)
val_loss_matrix   = np.array(val_loss_matrix)
acc_matrix        = np.array(acc_matrix)
clients = list(range(val_loss_matrix.shape[1]))  # Assuming all matrices have same shape

print(f"\n Metric matrices loaded | Train Loss: {train_loss_matrix.shape}, Val Loss: {val_loss_matrix.shape}, Accuracy: {acc_matrix.shape}")


# TRAIN vs VAL LOSS SURFACE
fig_loss = go.Figure()

fig_loss.add_trace(go.Surface(
    z=train_loss_matrix,
    x=clients,
    y=rounds,
    colorscale='Reds',
    name="Train Loss",
    showscale=True,
    colorbar=dict(title="Train Loss")
))

fig_loss.add_trace(go.Surface(
    z=val_loss_matrix,
    x=clients,
    y=rounds,
    colorscale='Blues',
    name="Val Loss",
    showscale=True,
    colorbar=dict(title="Val Loss"),
    opacity=0.9
))

fig_loss.update_layout(
    title="Train vs Val Loss Surface (Rounds x Clients)",
    scene=dict(
        xaxis_title='Client',
        yaxis_title='Round',
        zaxis_title='Loss',
    ),
    paper_bgcolor='lavenderblush'
)

fig_loss.show()

# Accuracy Surface Plot
fig_acc = go.Figure(data=[go.Surface(
    z=acc_matrix,
    x=clients,
    y=rounds,
    colorscale='YlGnBu',
    colorbar=dict(title='Accuracy'),
    showscale=True
)])

fig_acc.update_layout(
    title='Validation Accuracy Surface over Rounds & Clients',
    title_font_size=20,
    scene=dict(
        xaxis_title='Client',
        yaxis_title='Round',
        zaxis_title='Accuracy',
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    paper_bgcolor='lavenderblush',
    plot_bgcolor='honeydew',
)

fig_acc.show()

