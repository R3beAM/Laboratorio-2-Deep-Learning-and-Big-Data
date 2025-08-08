import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

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

# Define a simple neural network for regression
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def calc_metrics(preds: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Return MSE and R^2 in original target scale."""
    preds_np = preds.numpy() * y_std + y_mean
    target_np = target.numpy() * y_std + y_mean
    mse = mean_squared_error(target_np, preds_np)
    r2 = r2_score(target_np, preds_np)
    return mse, r2

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
            mse, r2 = calc_metrics(preds, tensor_y_test)
        print(
            f"Epoch {epoch+1}/{n_epochs}, Validation MSE: {mse:.4f}, R^2: {r2:.4f}"
        )

# Final evaluation
model.eval()
with torch.no_grad():
    preds = model(tensor_X_test)
    mse, r2 = calc_metrics(preds, tensor_y_test)

print(f"Final Test MSE: {mse:.2f}")
print(f"Final Test R^2: {r2:.4f}")
