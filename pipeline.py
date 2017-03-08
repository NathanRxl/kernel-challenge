from time import time
import numpy as np

import kernels
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
    y_train = data_loader.load_labels_only("Ytr.csv")

    # Compute the first train kernel
    X_train_1 = np.load("data/hog_grey_block_4_Xtr.npy")
    poly_kernel_parameters_1 = {
        "gamma": .01,
        "d": 6.,
        "r": 10.0,
    }
    K_train_1, K_diag_train_1 = kernels.polynomial_kernel_train(
        X_train_1, **poly_kernel_parameters_1, output_diag=True
    )

    # Compute the second train kernel
    X_train_2 = np.load("data/color_filter_window_5_Xtr.npy")
    X_train_2 = tools.normalize(X_train_2)
    poly_kernel_parameters_2 = {
        "gamma": .01,
        "d": 6.,
        "r": 2.0,
    }
    K_train_2, K_diag_train_2 = kernels.polynomial_kernel_train(
        X_train_2, **poly_kernel_parameters_2, output_diag=True
    )

    # Compute the third train kernel
    X_train_3 = np.load("data/color_filter_window_3_Xtr.npy")
    X_train_3 = tools.normalize(X_train_3)
    poly_kernel_parameters_3 = {
        "gamma": .01,
        "d": 6.,
        "r": 2.0,
    }
    K_train_3, K_diag_train_3 = kernels.polynomial_kernel_train(
        X_train_3, **poly_kernel_parameters_3, output_diag=True
    )

    # Combine the kernels
    alpha = 10. * 1e-4
    beta = 12. * 1e-4
    kernel = alpha * K_train_2 + (1 - alpha) * K_train_1 + beta * K_train_3
    K_diag_train = np.diag(kernel).reshape((5000,1))
    K_train = kernel / np.sqrt(K_diag_train.dot(K_diag_train.T))
else:
    features_file = "color_1_filter_Xtr.npy"
    X_train = np.load(path_to_data + features_file)
    y_train = data_loader.load_labels_only("Ytr.csv")


print("OK")

# Initiate the model
kernel_model = models.KernelLogisticRegression(
    penalty=kernel_ridge,
    grad_penalty=grad_kernel_ridge,
    lbda=0.1,
    multi_class="multinomial",
    kernel="precomputed"
)

print("\tFit the model to the train data ... ")
# Fit the model with the training data
if kernel_method:
    kernel_model.fit(K_train, y_train)
else:
    kernel_model.fit(X_train, y_train)

# # Compute the training score
# print("\tTraining score: ", end="", flush=True)
# if kernel_method:
#     y_predict_train = kernel_model.predict(K_train)
# else:
#     y_predict_train = kernel_model.predict(X_train)
# train_score = metrics.accuracy_score(y_predict_train, y_train)
# print(round(train_score, 5), end="\n\n")

print("\tLoad preprocessed test data ... ", end="", flush=True)

# Load test data
if kernel_method:
    ###############################
    # Compute the test/test kernels
    # Compute the first test/test kernel
    X_test_1 = np.load("data/hog_grey_block_4_Xte.npy")
    K_test_1 = kernels.polynomial_kernel_train(
        X_test_1, **poly_kernel_parameters_1
    )
    # Compute the second test/test kernel
    X_test_2 = np.load("data/color_filter_window_5_Xte.npy")
    X_test_2 = tools.normalize(X_test_2)
    K_test_2 = kernels.polynomial_kernel_train(
        X_test_2, **poly_kernel_parameters_2
    )
    # Compute the third test/test kernel
    X_test_3 = np.load("data/color_filter_window_3_Xte.npy")
    X_test_3 = tools.normalize(X_test_3)
    K_test_3 = kernels.polynomial_kernel_train(
        X_test_3, **poly_kernel_parameters_3
    )
    # Combine the kernels
    kernel = alpha * K_test_2 + (1 - alpha) * K_test_1 + beta * K_test_3
    K_diag_test_test = np.diag(kernel).reshape((2000,1))

    ###############################
    # Compute the test/train kernels
    K_test_1 = kernels.polynomial_kernel_test(
        X_test_1, X_train_1, K_diag_train_1, **poly_kernel_parameters_1
    )
    K_test_2 = kernels.polynomial_kernel_test(
        X_test_2, X_train_2, K_diag_train_2, **poly_kernel_parameters_2
    )
    K_test_3 = kernels.polynomial_kernel_test(
        X_test_3, X_train_3, K_diag_train_3, **poly_kernel_parameters_3
    )
    # Combine the kernels
    kernel = alpha * K_test_2 + (1 - alpha) * K_test_1 + beta * K_test_3
    K_test = kernel / np.sqrt(K_diag_test_test.dot(K_diag_train.T))
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

tools.create_submission(
    y_pred,
    submission_folder_path=submission_folder_path,
    output_filename="hog_4_bilateral_5_3.txt",
)
print("OK")


print("\nPipeline script completed in %0.2f seconds" % (time() - initial_time))
