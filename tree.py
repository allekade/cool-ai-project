import numpy as np
from first import load_phishing_data


def gini_impurity(y):
    """Compute Gini impurity for labels y."""
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob**2)

def best_split(X, y):
    best_feature, best_threshold = None, None
    best_gini = 1.0
    n_samples, n_features = X.shape

    for feature in range(n_features):
        values = np.sort(np.unique(X[:, feature]))
        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        for t in thresholds:
            left_idx = X[:, feature] <= t
            right_idx = np.logical_not(left_idx)

            if left_idx.sum() == 0 or right_idx.sum() == 0:
                continue

            g_left = gini_impurity(y[left_idx])
            g_right = gini_impurity(y[right_idx])

            g_total = (left_idx.sum() / n_samples) * g_left + \
                      (right_idx.sum() / n_samples) * g_right

            if g_total < best_gini:
                best_gini = g_total
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold



def build_tree(X, y, depth=0, max_depth=None):
    """Recursively build a decision tree."""

    if len(np.unique(y)) == 1:
        return {"leaf": True, "class": y[0]}

    if max_depth is not None and depth >= max_depth:
        classes, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": classes[np.argmax(counts)]}

    feature, threshold = best_split(X, y)

    if feature is None:
        classes, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": classes[np.argmax(counts)]}

    left_idx = X[:, feature] <= threshold
    right_idx = ~left_idx

    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[left_idx], y[left_idx], depth+1, max_depth),
        "right": build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    }

def tree_predict_one(tree, x):
    """Predict a single sample."""
    if tree["leaf"]:
        return tree["class"]

    if x[tree["feature"]] <= tree["threshold"]:
        return tree_predict_one(tree["left"], x)
    else:
        return tree_predict_one(tree["right"], x)

def tree_predict(tree, X):
    return np.array([tree_predict_one(tree, x) for x in X])

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

    max_depth = 29
    n_folds = 5

    accuracies = []

    for fold, (X_train, y_train, X_test, y_test) in enumerate(k_fold_split(X, y, K=n_folds)):
        
        tree = build_tree(X_train, y_train, max_depth=max_depth)
        
        y_pred = tree_predict(tree, X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}")

    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
