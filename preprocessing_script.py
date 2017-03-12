import pandas as pd
import tools
from time import time

initial_time = time()

print("Preprocessing script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

preprocess_color = True
preprocess_grey = True
preprocess_hist = False

print("\tLoad and preprocess train data ... ", end="", flush=True)

X_train = pd.read_csv(path_to_data + "Xtr.csv", header=None)
y_train = data_loader.load_labels_only("Ytr.csv")
color_X_train = X_train.dropna(axis=1, inplace=False)
if preprocess_grey or preprocess_hist:
    grey_X_train = tools.rgb2grey(color_X_train)
    if preprocess_hist:
        bins = 300
        hist_grey_X_train = tools.histogram_df(X_df=grey_X_train, bins=bins)
print("OK")

if preprocess_hist:
    preprocessed_train_filename = "hist_grey_{}_bins_Xtr.csv".format(bins)

    print(
        "\tCreate " + path_to_data + preprocessed_train_filename
        + " file ... ",
        end="",
        flush=True
    )
    hist_grey_X_train.to_csv(
        path_to_data + preprocessed_train_filename,
        index=False
    )

    print("OK")

if preprocess_grey:
    preprocessed_train_filename = "grey_Xtr.csv"

    print(
        "\tCreate " + path_to_data + preprocessed_train_filename + " file ... ",
        end="",
        flush=True
    )
    grey_X_train.to_csv(path_to_data + preprocessed_train_filename, index=False)

    print("OK")

if preprocess_color:
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

if preprocess_grey or preprocess_hist:
    grey_X_test = tools.rgb2grey(color_X_test)
    if preprocess_hist:
        bins = 300
        hist_grey_X_test = tools.histogram_df(X_df=grey_X_test, bins=bins)

print("OK")

if preprocess_hist:
    preprocessed_test_filename = "hist_grey_{}_bins_Xte.csv".format(bins)

    print(
        "\tCreate " + path_to_data + preprocessed_test_filename + " file ... ",
        end="",
        flush=True
    )
    hist_grey_X_test.to_csv(
        path_to_data + preprocessed_test_filename,
        index=False
    )
    print("OK")

if preprocess_grey:
    preprocessed_test_filename = "grey_Xte.csv"

    print(
        "\tCreate " + path_to_data + preprocessed_test_filename + " file ... ",
        end="",
        flush=True
    )
    grey_X_test.to_csv(path_to_data + preprocessed_test_filename, index=False)
    print("OK")

if preprocess_color:
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
