from time import time
import numpy as np

from sklearn.model_selection import KFold

import tools
import models
import metrics
import kernels
from penalizations import kernel_ridge, grad_kernel_ridge, ridge, grad_ridge

initial_time = time()

print("Cross-validation script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

kernel_method = True

if kernel_method:
    y_train = data_loader.load_labels_only("Ytr.csv")

    # Compute first kernel
    X_train_1 = np.load("data/hog_grey_block_4_Xtr.npy")
    poly_kernel_parameters_1 = {
        "gamma": .01,
        "d": 6.,
        "r": 10.0,
    }
    K_train_1 = kernels.polynomial_kernel_train(
        X_train_1, **poly_kernel_parameters_1
    )

    # Compute second kernel
    X_train_2 = np.load("data/color_filter_window_5_Xtr.npy")
    X_train_2 = tools.normalize(X_train_2)
    poly_kernel_parameters_2 = {
        "gamma": .01,
        "d": 6.,
        "r": 2.0,
    }
    K_train_2 = kernels.polynomial_kernel_train(
        X_train_2, **poly_kernel_parameters_2
    )

    # Compute third kernel
    X_train_3 = np.load("data/color_filter_window_3_Xtr.npy")
    X_train_3 = tools.normalize(X_train_3)
    poly_kernel_parameters_3 = {
        "gamma": .01,
        "d": 6.,
        "r": 2.0,
    }
    K_train_3 = kernels.polynomial_kernel_train(
        X_train_3, **poly_kernel_parameters_3
    )

    # Combine the kernels
    alpha = 10. * 1e-4
    beta = 12. * 1e-4
    kernel = alpha * K_train_2 + (1 - alpha) * K_train_1 + beta * K_train_3
    K_diag = np.diag(kernel).reshape((5000,1))
    K_train = kernel / np.sqrt(K_diag.dot(K_diag.T))

else:
    features_file = "color_1_filter_Xtr.npy"
    X_train = np.load(path_to_data + features_file)
    y_train = data_loader.load_labels_only("Ytr.csv")
    print("Number of features :", X_train.shape[1], end="\n\n")


# Cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
cv_scores = []

if kernel_method:
    cv_splits = kf.split(np.arange(len(K_train)))
else:
    cv_splits = kf.split(np.arange(len(X_train)))

# Define the model
kernel_model = models.KernelLogisticRegression(
    penalty=kernel_ridge,
    grad_penalty=grad_kernel_ridge,
    kernel="precomputed",
    lbda=.5,
    multi_class="multinomial",
)

for n_fold, (train_fold_idx, test_fold_idx) in enumerate(cv_splits):

    print(
        "Start working on fold number", n_fold + 1, "... ",
        end="",
        flush=True
    )

    if kernel_method:
        K_fold_train = K_train[train_fold_idx, :][:, train_fold_idx]
    else:
        X_fold_train = X_train[train_fold_idx]
    y_fold_train = y_train[train_fold_idx]

    if kernel_method:
        K_fold_test = K_train[test_fold_idx, :][:, train_fold_idx]
    else:
        X_fold_test = X_train[test_fold_idx]
    y_fold_test = y_train[test_fold_idx]

    if kernel_method:
        kernel_model.fit(K_fold_train, y_fold_train)
        y_pred = kernel_model.predict(K_fold_test)
    else:
        kernel_model.fit(X_fold_train, y_fold_train)
        y_pred = kernel_model.predict(X_fold_test)

    fold_score = metrics.accuracy_score(y_pred, y_fold_test)

    print(fold_score)
    cv_scores.append(fold_score)

cv_score_mean = round(np.mean(cv_scores), 5)
cv_score_std = round(np.std(cv_scores), 5)
cv_score_min = round(np.min(cv_scores), 5)
cv_score_max = round(np.max(cv_scores), 5)

print("\nMean score :", cv_score_mean)
print("Standard deviation :", cv_score_std)
print("Min score :", cv_score_min)
print("Max score :", cv_score_max)


print("\nCross-validation script completed in %0.2f seconds"
      % (time() - initial_time))
