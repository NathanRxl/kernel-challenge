from time import time
import numpy as np

import tools
import models
import metrics
from penalizations import ridge, grad_ridge

initial_time = time()

print("Pipeline script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

print("\tLoad preprocessed train data ... ", end="", flush=True)
X_train, y_train = data_loader.load_data("grey_Xtr.csv", "Ytr.csv")

features_file = "color_1_filter_Xtr.npy"
X_train = np.load(path_to_data + features_file)
print("OK")

# Initiate the model
kernel_model = models.LogisticRegression(
    penalty=ridge,
    grad_penalty=grad_ridge,
    lbda=0.1,
    multi_class="multinomial"
)

print("\tFit the model to the train data ... ")
# Fit the model with the training data
kernel_model.fit(X_train, y_train)

# Compute the training score
print("\tTraining score: ", end="", flush=True)
y_predict_train = kernel_model.predict(X_train)
train_score = metrics.accuracy_score(y_predict_train, y_train)
print(round(train_score, 5), end="\n\n")

print("\tLoad preprocessed test data ... ", end="", flush=True)
# Load test data
X_test = data_loader.load_data("grey_Xte.csv")

features_file = "color_1_filter_Xte.npy"
X_test = np.load(path_to_data + features_file)
print("OK")

print("\tMake predictions on test data ... ", end="", flush=True)
# Predict the labels of X_test
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
