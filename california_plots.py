import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Load California housing data
# Columns: longitude, latitude, housing_median_age, total_rooms,
# total_bedrooms, population, households, median_income, median_house_value

data_path = 'CaliforniaHousing/cal_housing.data'
data = np.loadtxt(data_path, delimiter=',')

X = data[:, :-1]
y = data[:, -1:]

# Train-test split
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
split = int(len(X) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Standardize features and target
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# Convert to tensors
tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define neural network
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with history tracking
n_epochs = 20
train_losses = []
val_losses = []
for epoch in range(n_epochs):
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

    model.eval()
    with torch.no_grad():
        val_pred = model(tensor_X_test)
        val_loss = criterion(val_pred, tensor_y_test).item()
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Train MSE: {train_losses[-1]:.4f}, Validation MSE: {val_loss:.4f}")

# Final predictions for plotting
model.eval()
with torch.no_grad():
    preds = model(tensor_X_test)
    preds_orig = preds.numpy() * y_std + y_mean
    y_test_orig = tensor_y_test.numpy() * y_std + y_mean

# Plot training vs validation MSE
plt.figure()
plt.plot(range(1, n_epochs + 1), train_losses, label='Train MSE')
plt.plot(range(1, n_epochs + 1), val_losses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation MSE')
plt.legend()
plt.tight_layout()
plt.savefig('mse_plot.png')

# Plot predicted vs actual values
plt.figure()
plt.scatter(y_test_orig, preds_orig, alpha=0.5)
min_val = min(y_test_orig.min(), preds_orig.min())
max_val = max(y_test_orig.max(), preds_orig.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('pred_vs_actual.png')

print('Plots saved as mse_plot.png and pred_vs_actual.png')
