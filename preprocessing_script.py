import pandas as pd
import tools
from time import time

initial_time = time()

print("Preprocessing script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

print("\tLoad and preprocess train data ... ", end="", flush=True)

X_train = pd.read_csv(path_to_data + "Xtr.csv", header=None)
y_train = data_loader.load_labels_only("Ytr.csv")
color_X_train = X_train.dropna(axis=1, inplace=False)
grey_X_train = tools.rgb2grey(color_X_train)

print("OK")


preprocessed_train_filename = "grey_Xtr.csv"

print(
    "\tCreate " + path_to_data + preprocessed_train_filename + " file ... ",
    end="",
    flush=True
)
grey_X_train.to_csv(path_to_data + preprocessed_train_filename, index=False)

print("OK")

preprocessed_train_filename = "color_Xtr.csv"

print(
    "\tCreate " + path_to_data + preprocessed_train_filename + " file ... ",
    end="",
    flush=True
)
color_X_train.to_csv(path_to_data + preprocessed_train_filename, index=False)

print("OK", end="\n\n")


print("\tLoad and preprocess test data ... ", end="", flush=True)

X_test = pd.read_csv(path_to_data + "Xte.csv", header=None)
color_X_test = X_test.dropna(axis=1, inplace=False)
grey_X_test = tools.rgb2grey(color_X_test)

print("OK")

preprocessed_test_filename = "grey_Xte.csv"

print(
    "\tCreate " + path_to_data + preprocessed_test_filename + " file ... ",
    end="",
    flush=True
)
grey_X_test.to_csv(path_to_data + preprocessed_test_filename, index=False)
print("OK")

preprocessed_train_filename = "color_Xte.csv"

print(
    "\tCreate " + path_to_data + preprocessed_train_filename + " file ... ",
    end="",
    flush=True
)
color_X_test.to_csv(path_to_data + preprocessed_train_filename, index=False)

print("OK", end="\n\n")


print(
    "\nPreprocessing script completed in %0.2f seconds"
    % (time() - initial_time)
)
