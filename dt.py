class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        predictions = [self._predict(x, self.tree) for x in X]
        return predictions

    def _gini(self, y):
        classes = np.unique(y)
        m = y.size
        gini = 1.0
        for c in classes:
            p_c = np.count_nonzero(y == c) / m
            gini -= p_c ** 2
        return gini

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        m, n = X.shape
        best_gini = float('inf')
        best_split = None
        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                gini = (y_left.size / m) * gini_left + (y_right.size / m) * gini_right
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)
        return best_split

    def _build_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            feature_index, threshold = self._best_split(X, y)
            if feature_index is not None:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                node['feature_index'] = feature_index
                node['threshold'] = threshold
                node['left'] = self._build_tree(X_left, y_left, depth + 1)
                node['right'] = self._build_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, x, node):
        if 'predicted_class' in node:
            return node['predicted_class']
        if x[node['feature_index']] <= node['threshold']:
            return self._predict(x, node['left'])
        else:
            return self._predict(x, node['right'])
