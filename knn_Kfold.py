import numpy as np
from first import load_phishing_data

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) **2))

def knn_predict(X_train, y_train, X_test, k=5):
    #knn implmentation here

    return np.array(predictions)

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

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    K = 5 #amount of neigbors

    n_folds = 5 #folds for testing

    accuracies = []
    for fold, (X_train, y_train, X_test, y_test) in enumerate(k_fold_split(X, y, K=n_folds)):
        y_pred = knn_predict(X_train, y_train, X_test, K=K)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}")

    print("\nAverage Accuracy:", np.mean(accuracies))
    print("Standard Deviation:", np.std(accuracies))
