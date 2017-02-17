from time import time
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import tools
import models

initial_time = time()

print("Cross-validation script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)
X_train, y_train = data_loader.load_data("grey_Xtr.csv", "Ytr.csv")

# Cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
cv_scores = []
cv_splits = kf.split(X_train.index.tolist())

# Define the model
kernel_model = models.LogisticRegression(
        C=0.03,
        multi_class='multinomial',
        solver='lbfgs',
        n_jobs=-1
)

for n_fold, (train_fold_idx, test_fold_idx) in enumerate(cv_splits):
    
    print(
        "Start working on fold number", n_fold + 1, "... ",
        end="",
        flush=True
    )
    
    X_fold_train = X_train.iloc[train_fold_idx]
    y_fold_train = y_train[train_fold_idx]
    
    X_fold_test = X_train.iloc[test_fold_idx]
    y_fold_test = y_train[test_fold_idx]

    kernel_model.fit(X_fold_train, y_fold_train)
    y_pred = kernel_model.predict(X_fold_test)
    
    fold_score = accuracy_score(y_pred, y_fold_test)
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
