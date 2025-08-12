


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



if __name__ == '__main__':
    main()
