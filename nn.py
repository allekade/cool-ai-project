import numpy as np
from first import load_phishing_data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def initialize_weights(n_input, n_hidden):
    np.random.seed(42)
    W1 = np.random.randn(n_input, n_hidden) * 0.1
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * 0.1
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backward(X, y, z1, a1, z2, a2, W1, b1, W2, b2, lr):
    m = len(X)
    y = y.reshape(-1, 1)

    dz2 = a2 - y
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = (dz2 @ W2.T) * (1 - np.tanh(z1)**2)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return W1, b1, W2, b2

def train_network(X, y, hidden=16, epochs=250, lr=0.05):
    n_input = X.shape[1]
    W1, b1, W2, b2 = initialize_weights(n_input, hidden)

    for _ in range(epochs):
        z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward(X, y, z1, a1, z2, a2,
                                  W1, b1, W2, b2, lr)

    return W1, b1, W2, b2

def nn_predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward(X, W1, b1, W2, b2)
    return (a2 > 0.5).astype(int).flatten()


def k_fold_split(X, y, K=4):
    fold_size = len(X) // K
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for i in range(K):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
    X, y = load_phishing_data()

    y = np.where(y == -1, 0, y)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    n_folds = 5
    accuracies = []

    for fold, (X_train, y_train, X_test, y_test) in enumerate(k_fold_split(X, y, K=n_folds)):
        W1, b1, W2, b2 = train_network(X_train, y_train,
                                       hidden=16, epochs=250, lr=0.05)

        y_pred = nn_predict(X_test, W1, b1, W2, b2)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold+1}: Accuracy = {acc:.4f}")

    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
