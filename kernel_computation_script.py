from time import time

import tools
import kernels

initial_time = time()

print("Kernel computation script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

compute_linear_kernel = True
compute_intersection_kernel = True

if compute_linear_kernel:
    print("Compute linear kernel")

    preprocessed_train_filename = "grey_Xtr.csv"
    print(
        "\tLoad {} ... ".format(preprocessed_train_filename),
        end="",
        flush=True
    )
    grey_X_train = (
        data_loader.load_data(preprocessed_train_filename)
    )
    print("OK")

    print(
        "\tCompute linear Kernel on {} ... "
        .format(preprocessed_train_filename),
        end="",
        flush=True
    )
    K_train = kernels.linear_kernel(grey_X_train, grey_X_train)
    print("OK")

    K_train_filename = "grey_Ktr.csv"
    print(
        "\tStore linear Kernel in {} ... "
        .format(K_train_filename),
        end="",
        flush=True
    )
    K_train.tofile(path_to_data + K_train_filename)
    print("OK", end="\n\n")

    preprocessed_test_filename = "grey_Xte.csv"
    print(
        "\tLoad {} ... ".format(preprocessed_test_filename),
        end="",
        flush=True
    )
    grey_X_test = data_loader.load_data(preprocessed_test_filename)
    print("OK")

    print(
        "\tCompute linear Kernel on {} ... "
        .format(preprocessed_test_filename),
        end="",
        flush=True
    )
    K_test = kernels.linear_kernel(grey_X_test, grey_X_train)
    print("OK")

    K_test_filename = "grey_Kte.csv"
    print(
        "\tStore linear Kernel in {} ... "
        .format(K_test_filename),
        end="",
        flush=True
    )
    K_test.tofile(path_to_data + K_test_filename)
    print("OK", end="\n\n")

if compute_intersection_kernel:
    print("Compute intersection kernel")
    bins = 300

    preprocessed_train_filename = "hist_grey_{}_bins_Xtr.csv".format(bins)
    print(
        "\tLoad {} ... ".format(preprocessed_train_filename),
        end="",
        flush=True
    )
    hist_grey_X_train = (
        data_loader.load_data(preprocessed_train_filename)
    )
    print("OK")

    print(
        "\tCompute intersection Kernel on {} ... "
        .format(preprocessed_train_filename),
        end="",
        flush=True
    )
    K_train = kernels.intersection_kernel(hist_grey_X_train, hist_grey_X_train)
    print("OK")

    K_train_filename = "hist_grey_{}_bins_intersection_Ktr.csv".format(bins)
    print(
        "\tStore intersection Kernel in {} ... "
        .format(K_train_filename),
        end="",
        flush=True
    )
    K_train.tofile(path_to_data + K_train_filename)
    print("OK", end="\n\n")

    preprocessed_test_filename = "hist_grey_{}_bins_Xte.csv".format(bins)
    print(
        "\tLoad {} ... ".format(preprocessed_test_filename),
        end="",
        flush=True
    )
    hist_grey_X_test = (
        data_loader.load_data(preprocessed_test_filename)
    )
    print("OK")

    print(
        "\tCompute intersection Kernel on {} ... "
        .format(preprocessed_test_filename),
        end="",
        flush=True
    )
    K_test = kernels.intersection_kernel(hist_grey_X_test, hist_grey_X_train)
    print("OK")

    K_test_filename = "hist_grey_{}_bins_intersection_Kte.csv".format(bins)
    print(
        "\tStore intersection Kernel in {} ... "
        .format(K_test_filename),
        end="",
        flush=True
    )
    K_test.tofile(path_to_data + K_test_filename)
    print("OK", end="\n\n")

print(
    "\nKernel computation script completed in %0.2f seconds"
    % (time() - initial_time)
)
