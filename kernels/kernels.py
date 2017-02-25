from itertools import product
import numpy as np


def intersection_kernel(hist_test, hist_train):
    hist_test_len = len(hist_test)
    hist_train_len = len(hist_train)
    K = np.zeros(shape=(hist_test_len, hist_train_len))
    if hist_test_len == hist_train_len:
        for i, j in product(range(hist_test_len), range(hist_train_len)):
            if i <= j:
                K[i, j] = np.minimum(hist_test[i, :], hist_train[j, :]).sum()
        return K + np.triu(K, k=1).T
    else:
        for i, j in product(range(hist_test_len), range(hist_train_len)):
            K[i, j] = np.minimum(hist_test[i, :], hist_train[j, :]).sum()
        return K


def linear_kernel(X_test, X_train):
    return X_test.dot(X_train.T)
