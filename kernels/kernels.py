from itertools import product
import numpy as np


def normalize_kernel(K):
    normalized_K = np.zeros_like(K)
    n_rows_K, n_columns_K = K.shape
    for i, j in product(range(n_rows_K), range(n_columns_K)):
        if i <= j:
            normalized_K[i, j] = K[i, j] / (np.sqrt(K[i, i]) * np.sqrt(K[j, j]))
    return normalized_K + np.triu(normalized_K, k=1).T


def intersection_kernel(hist_test, hist_train, degree=1):
    hist_test_len = len(hist_test)
    hist_train_len = len(hist_train)
    if hist_test_len == hist_train_len:
        K = np.zeros(shape=(hist_test_len, hist_train_len))
        for i, j in product(range(hist_test_len), range(hist_train_len)):
            if i <= j:
                K[i, j] = np.minimum(
                    np.power(hist_test[i, :], degree),
                    np.power(hist_train[j, :], degree)
                ).sum()
        return normalize_kernel(K + np.triu(K, k=1).T)
    else:
        # Make the kernel symmetric first to normalize it and then crop it
        hist_len_sum = hist_test_len + hist_train_len
        K = np.zeros(shape=(hist_len_sum, hist_len_sum))
        for i, j in product(range(hist_len_sum), range(hist_len_sum)):
            if i <= j:
                # K has X_test + X_train lines and X_test + X_train columns
                # It is a block matrix Ktest, Ktesttrain | Ktraintest, Ktrain
                if i < hist_test_len and j < hist_test_len:
                    K[i, j] = np.minimum(
                        np.power(hist_test[i, :], degree),
                        np.power(hist_test[j, :], degree)
                    ).sum()
                elif i < hist_test_len and hist_test_len <= j < hist_len_sum:
                    K[i, j] = np.minimum(
                        np.power(hist_test[i, :], degree),
                        np.power(hist_train[j - hist_test_len, :], degree)
                    ).sum()
                else:
                    K[i, j] = np.minimum(
                        np.power(hist_train[i - hist_test_len, :], degree),
                        np.power(hist_train[j - hist_test_len, :], degree)
                    ).sum()
        normalized_K = normalize_kernel(K + np.triu(K, k=1).T)
        return (
            normalized_K
            [np.arange(0, hist_test_len), :]
            [:, np.arange(hist_test_len, hist_len_sum)]
        )


def linear_kernel(X_test, X_train):
    X_test_len = len(X_test)
    X_train_len = len(X_train)
    if X_test_len == X_train_len:
        K = X_test.dot(X_train.T)
        return normalize_kernel(K)
    else:
        K = (np.asarray(
                np.bmat([
                    [X_test.dot(X_test.T), X_test.dot(X_train.T)],
                    [X_train.dot(X_test.T), X_train.dot(X_train.T)]
                ])
            ))
        normalized_K = normalize_kernel(K)
        return (
            normalized_K
            [np.arange(0, X_test_len), :]
            [:, np.arange(X_test_len, X_train_len + X_test_len)]
        )


def polynomial_kernel_test(X_test, X_train, K_diag_train, d, r, gamma):
    kernel = np.power(gamma * X_test.dot(X_train.T) + r, d)

    test_kernel = np.power(gamma * X_test.dot(X_test.T) + r, d)
    K_test_diag = np.diag(test_kernel).reshape((X_test.shape[0],1))

    return kernel / np.sqrt(K_test_diag.dot(K_diag_train.T))


def polynomial_kernel_train(X_train, d, r, gamma, output_diag=False):
    kernel = np.power(gamma * X_train.dot(X_train.T) + r, d)
    K_diag = np.diag(kernel).reshape((X_train.shape[0],1))

    if output_diag:
        return kernel / np.sqrt(K_diag.dot(K_diag.T)), K_diag
    else:
        return kernel / np.sqrt(K_diag.dot(K_diag.T))
