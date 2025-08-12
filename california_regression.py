import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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

# Define a small feedforward network for regression.
# Two hidden layers (64, 32) with ReLU activations provide enough capacity to learn
# non-linear interactions in the eight input features while keeping the parameter
# count manageable to limit overfitting.
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(tensor_X_test)
            val_loss = criterion(preds, tensor_y_test)
        print(f"Epoch {epoch+1}/{n_epochs}, Validation MSE: {val_loss.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    preds = model(tensor_X_test)
    # Convert predictions and targets back to original scale
    preds_orig = preds.numpy() * y_std + y_mean
    y_test_orig = tensor_y_test.numpy() * y_std + y_mean
    mse = np.mean((preds_orig - y_test_orig) ** 2)
    ss_res = ((preds_orig - y_test_orig) ** 2).sum()
    ss_tot = ((y_test_orig - y_test_orig.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

print(f"Final Test MSE: {mse:.2f}")
print(f"Final Test R^2: {r2:.4f}")