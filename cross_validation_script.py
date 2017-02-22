from time import time
import numpy as np

from sklearn.model_selection import KFold

import tools
import models
import metrics
from penalizations import ridge, grad_ridge

initial_time = time()

print("Cross-validation script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

# Load and compute the features
X_train, y_train = data_loader.load_data("color_Xtr.csv", "Ytr.csv")

features_file = "color_1_filter_Xtr.npy"
X_train = np.load(path_to_data + features_file)

print("Number of features :", X_train.shape[1], end="\n\n")

# Cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
cv_scores = []
cv_splits = kf.split(np.arange(len(y_train)))

# Define the model
kernel_model = models.LogisticRegression(
    penalty=ridge,
    grad_penalty=grad_ridge,
    lbda=0.1,
    multi_class="multinomial"
)


for n_fold, (train_fold_idx, test_fold_idx) in enumerate(cv_splits):

    print(
        "Start working on fold number", n_fold + 1, "... ",
        end="",
        flush=True
    )

    X_fold_train = X_train[train_fold_idx]
    y_fold_train = y_train[train_fold_idx]

    X_fold_test = X_train[test_fold_idx]
    y_fold_test = y_train[test_fold_idx]

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
