import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess(path: str):
    """Cargar datos, dividirlos en entrenamiento/prueba y estandarizar."""
    # 1) Carga de los datos desde el archivo .data
    data = np.loadtxt(path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1:]

    # 2) División 80/20 utilizando una permutación fija para reproducibilidad
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 3) Estandarización de características y objetivo
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # 4) Conversión a tensores y creación del DataLoader
    tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
    tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_dataset, test_tensors, y_mean, y_std


def build_model(input_dim, hidden_layers):
    """Create a feedforward network with given hidden layer sizes."""
    layers = []
    prev_dim = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


def train_and_evaluate(train_dataset, test_tensors, y_mean, y_std,
                        hidden_layers, loss_name, opt_name, lr,
                        epochs=50, batch_size=64):
    """Train a model and compute test MSE and R^2."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_dim = train_dataset.tensors[0].shape[1]
    model = build_model(input_dim, hidden_layers)

    if loss_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss_name == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss {loss_name}")

    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    X_test, y_test = test_tensors
    with torch.no_grad():
        preds = model(X_test)
        preds_orig = preds.numpy() * y_std + y_mean
        y_orig = y_test.numpy() * y_std + y_mean
        mse = np.mean((preds_orig - y_orig) ** 2)
        ss_res = ((preds_orig - y_orig) ** 2).sum()
        ss_tot = ((y_orig - y_orig.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
    return mse, r2


def main():
    torch.manual_seed(0)
    train_dataset, test_tensors, y_mean, y_std = load_data()

    epochs = int(os.environ.get("EPOCHS", 50))
    batch_size = 64
    hidden_configs = {'[64]': [64], '[16, 8]': [16, 8]}
    losses = ['MSELoss', 'SmoothL1Loss']
    optimizers = {'Adam': 0.001, 'SGD': 0.01}

    results = []
    for loss_name in losses:
        for opt_name, lr in optimizers.items():
            for cfg_name, layers in hidden_configs.items():
                mse, r2 = train_and_evaluate(
                    train_dataset, test_tensors, y_mean, y_std,
                    hidden_layers=layers, loss_name=loss_name,
                    opt_name=opt_name, lr=lr, epochs=epochs,
                    batch_size=batch_size)
                results.append({
                    'loss': loss_name,
                    'optimizer': opt_name,
                    'layers': cfg_name,
                    'lr': lr,
                    'epochs': epochs,
                    'batch': batch_size,
                    'mse': mse,
                    'r2': r2,
                })

    print("Experimento | Pérdida | Optimizador | Capas ocultas | LR | Épocas | Batch | MSE test | R² test")
    for i, exp in enumerate(results, 1):
        print(f"{i:>10} | {exp['loss']:^8} | {exp['optimizer']:^10} | {exp['layers']:^13} | "
              f"{exp['lr']:<5} | {exp['epochs']:^6} | {exp['batch']:^5} | "
              f"{exp['mse']:.2e} | {exp['r2']:.3f}")


if __name__ == '__main__':
    main()
