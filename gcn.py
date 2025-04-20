import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import haversine_distances
import torch.nn.functional as F

# Device setup
CUDA = True
device = "cuda" if (torch.cuda.is_available() and CUDA) else "cpu"

# Load data
node_df = pd.read_csv("data/node_features.csv")
road_gdf = gpd.read_file("data/singapore_road_network.gpkg", layer="edges")

# Graph Construction
node_df = node_df.drop(columns=[col for col in node_df.columns if "Unnamed" in col])
node_df = node_df.dropna()

coords = node_df[['Latitude', 'Longitude']].to_numpy()
coords_rad = np.radians(coords)
distances = haversine_distances(coords_rad) * 6371000  # in meters

# Define connectivity threshold (in meters)
edge_index = np.array(np.where((distances > 0) & (distances < 1000)))
edge_attr = distances[edge_index[0], edge_index[1]]

# Node features
node_df = node_df.rename(columns={
    'Total Casualties Fatalities': 'Total_Casualties_Fatalities',
    'Pedestrians': 'Pedestrians',
    'Rainfall_mm': 'Rainfall',
    'Traffic_Volume': 'Traffic_Volume'
})

feature_cols = ['Rainfall', 'Traffic_Volume', 'Total_Casualties_Fatalities', 'Pedestrians']
node_df.columns = node_df.columns.str.strip()
features = node_df[feature_cols].astype(float).to_numpy()
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(features), dtype=torch.float)

# Binary labels: hotspot or not
threshold = node_df['Total_Casualties_Fatalities'].astype(float).median()
y = torch.tensor((node_df['Total_Casualties_Fatalities'].astype(float) > threshold).astype(int).values, dtype=torch.float)

data = Data(
    x=x,
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    y=y
)

# Train-Test-Validation Split
indices = np.arange(data.num_nodes)
train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42, stratify=y)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # Output single logit per node

model = GCN(in_channels=x.size(1), hidden_channels=16, out_channels=1).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Training Loop
train_losses, val_accuracies = [], []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).squeeze()  # Shape: [num_nodes]
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    val_out = torch.sigmoid(out[data.val_mask])  # Apply sigmoid for probabilities
    val_pred = (val_out > 0.5).float()  # Threshold at 0.5
    val_acc = accuracy_score(data.y[data.val_mask].cpu(), val_pred.cpu())

    train_losses.append(loss.item())
    val_accuracies.append(val_acc)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

# Evaluation on Test Set
model.eval()
test_out = model(data.x, data.edge_index).squeeze()
test_pred = (torch.sigmoid(test_out[data.test_mask]) > 0.5).float()
test_acc = accuracy_score(data.y[data.test_mask].cpu(), test_pred.cpu())
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(data.y[data.test_mask].cpu(), test_pred.cpu())
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (GCN)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Hotspot', 'Hotspot'])
plt.yticks(tick_marks, ['Non-Hotspot', 'Hotspot'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix_gcn.png')
plt.close()

# Feature Importance with GNNExplainer
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='raw',  # Logits (single value per node)
    ),
)

node_idx = int(val_idx[0])
explanation = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
importance = explanation.node_mask.detach().cpu().numpy().flatten()
num_features = 4
start = node_idx * num_features
end = start + num_features
importance = importance[start:end]

feature_labels = ['Rainfall', 'Traffic Volume', 'Total Casualties', 'Pedestrians']
indices = np.argsort(importance)[::-1]
sorted_importance = importance[indices]
sorted_labels = [feature_labels[int(i)] for i in indices]

plt.figure(figsize=(8, 6))
bars = plt.barh(range(len(sorted_labels)), sorted_importance, color='skyblue')
plt.yticks(range(len(sorted_labels)), sorted_labels)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title(f"Feature Importance for Node {node_idx} (GCN)")
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.01, bar.get_y() + 0.3, f"{sorted_importance[i]:.3f}")
plt.tight_layout()
plt.savefig('feature_importance_gcn.png')
plt.close()