"""Paso a paso para entrenar y evaluar una red neuronal en el conjunto California Housing."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


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

    return train_loader, tensor_X_test, tensor_y_test, (y_mean, y_std)


def build_model(input_dim: int) -> nn.Module:
    """Definir la arquitectura totalmente conectada."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def train(model: nn.Module, loader: DataLoader, criterion, optimizer, epochs: int = 20):
    """Entrenar el modelo e imprimir el MSE de entrenamiento cada 5 épocas."""
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Época {epoch+1}/{epochs} - MSE entrenamiento: {loss.item():.4f}")


def evaluate(model: nn.Module, X_test, y_test, y_scale):
    """Calcular el MSE y R² en el conjunto de prueba."""
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds_orig = preds.numpy() * y_scale[1] + y_scale[0]
        y_test_orig = y_test.numpy() * y_scale[1] + y_scale[0]
        mse = np.mean((preds_orig - y_test_orig) ** 2)
        ss_res = ((preds_orig - y_test_orig) ** 2).sum()
        ss_tot = ((y_test_orig - y_test_orig.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
    print(f"MSE de prueba: {mse:.2f}")
    print(f"R² de prueba: {r2:.4f}")


def main():
    print("Paso 1: cargar y preprocesar datos")
    train_loader, X_test, y_test, y_scale = load_and_preprocess('CaliforniaHousing/cal_housing.data')

    print("Paso 2: definir el modelo y los hiperparámetros")
    model = build_model(X_test.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Paso 3: entrenar el modelo")
    train(model, train_loader, criterion, optimizer)

    print("Paso 4: evaluar el modelo")
    evaluate(model, X_test, y_test, y_scale)


if __name__ == '__main__':
    main()
