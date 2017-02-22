import pandas as pd
import numpy as np
import tools
from time import time

initial_time = time()

print("Filtering script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

# Filtering parameters to apply on train and test images
filtering_parameters_list = [
    {"window_size": 5, "sigma": 10},
]
n_filters = len(filtering_parameters_list)
print("\tWe will apply {n_filters} filter to the train and test images."
      .format(n_filters=n_filters))


################################################################################
# Filter train images
print("\tLoad and filter train images ... ")

X_train, y_train = data_loader.load_data("color_Xtr.csv", "Ytr.csv")
color_train_images = tools.color_images_from_df(X_train)

features_list = [
    tools.bilateral_filter(color_train_images, **filtering_parameters)
    for filtering_parameters in filtering_parameters_list
]

filtered_X_train = np.concatenate(features_list, axis=1)
print("\tOK")


filtered_train_filename = "color_" + str(n_filters) + "_filter_Xtr"
print(
    "\tCreate " + path_to_data + filtered_train_filename + ".npy file ... ",
    end="",
    flush=True
)
np.save(path_to_data + filtered_train_filename, filtered_X_train)
print("OK")


################################################################################
# Filter test images
print("\n\tLoad and filter test images ... ")

X_test = data_loader.load_data("color_Xte.csv")
color_test_images = tools.color_images_from_df(X_test)

features_list = [
    tools.bilateral_filter(color_test_images, **filtering_parameters)
    for filtering_parameters in filtering_parameters_list
]

filtered_X_test = np.concatenate(features_list, axis=1)
print("\tOK")


filtered_test_filename = "color_" + str(n_filters) + "_filter_Xte"
print(
    "\tCreate " + path_to_data + filtered_test_filename + ".npy file ... ",
    end="",
    flush=True
)
np.save(path_to_data + filtered_test_filename, filtered_X_test)
print("OK")


print("\nFiltering script completed in %0.2f seconds"% (time() - initial_time))
