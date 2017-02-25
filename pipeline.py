from time import time
import numpy as np

import tools
import models
import metrics
from penalizations import kernel_ridge, grad_kernel_ridge

initial_time = time()

print("Pipeline script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)
kernel_method = True

print("\tLoad preprocessed train data ... ", end="", flush=True)

if kernel_method:
    K_train, y_train = data_loader.load_kernel("grey_Ktr.csv", "Ytr.csv")
else:
    features_file = "color_1_filter_Xtr.npy"
    X_train = np.load(path_to_data + features_file)
    y_train = data_loader.load_labels_only("Ytr.csv")

print("OK")

# Initiate the model
kernel_model = models.KernelLogisticRegression(
    penalty=kernel_ridge,
    grad_penalty=grad_kernel_ridge,
    lbda=5,
    multi_class="multinomial",
    kernel="precomputed"
)

print("\tFit the model to the train data ... ")
# Fit the model with the training data
if kernel_method:
    kernel_model.fit(K_train, y_train)
else:
    kernel_model.fit(X_train, y_train)

# Compute the training score
print("\tTraining score: ", end="", flush=True)
if kernel_method:
    y_predict_train = kernel_model.predict(K_train)
else:
    y_predict_train = kernel_model.predict(X_train)
train_score = metrics.accuracy_score(y_predict_train, y_train)
print(round(train_score, 5), end="\n\n")

print("\tLoad preprocessed test data ... ", end="", flush=True)

# Load test data
if kernel_method:
    K_test = data_loader.load_kernel(
        "grey_Kte.csv",
        train_samples=len(K_train)
    )
else:
    features_file = "color_1_filter_Xte.npy"
    X_test = np.load(path_to_data + features_file)

print("OK")

print("\tMake predictions on test data ... ", end="", flush=True)
# Predict the labels of X_test
if kernel_method:
    y_pred = kernel_model.predict(K_test)
else:
    y_pred = kernel_model.predict(X_test)
print("OK", end="\n\n")


print(
    "\tCreate Kaggle submission in submissions/ folder ... ",
    end="",
    flush=True
)
# Create Kaggle submission
submission_folder_path = "submissions/"

tools.create_submission(y_pred, submission_folder_path=submission_folder_path)
print("OK")


print("\nPipeline script completed in %0.2f seconds" % (time() - initial_time))
