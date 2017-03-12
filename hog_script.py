import sys
import numpy as np
from time import time

import tools


initial_time = time()
print("Gradient histogram script", end="\n\n")

path_to_data = "data/"
data_loader = tools.DataLoader(path_to_data=path_to_data)

if len(sys.argv) < 2:
    cell_size = 16
    block_size = 1
else:
    cell_size = int(sys.argv[1])
    block_size = int(sys.argv[2])

hog_parameters = {
    "n_orientations": 8,
    "pixels_per_cell": (cell_size, cell_size),
    "cells_per_block": (block_size, block_size),
    "method": "L1",
}

################################################################################
# Filter train images
print("\tLoad and filter train images ... ")
train_images = data_loader.load_data("grey_Xtr.csv")
n_images = train_images.shape[0]
std = np.std(train_images, axis=0)

train_images = train_images / std
train_images = train_images.reshape((n_images,32,32))

train_hog = tools.hog_filter(train_images, **hog_parameters)
train_hog = (train_hog - np.mean(train_hog, axis=0)) / np.std(train_hog, axis=0)
print("\tOK")

output_filename = "hog_grey_cell_{cell}_block_{block}_Xtr"\
                  .format(cell=cell_size, block=block_size)
print(
    "\tCreate " + path_to_data + output_filename + ".npy file ... ",
    end="",
    flush=True
)
np.save(path_to_data + output_filename, train_hog)
print("OK")

################################################################################
# Filter test images
print("\n\tLoad and filter train images ... ")
test_images = data_loader.load_data("grey_Xte.csv")
n_images = test_images.shape[0]
std = np.std(test_images, axis=0)

test_images = test_images / std
test_images = test_images.reshape((n_images,32,32))

test_hog = tools.hog_filter(test_images, **hog_parameters)
test_hog = (test_hog - np.mean(test_hog, axis=0)) / np.std(test_hog, axis=0)
print("\tOK")

output_filename = "hog_grey_cell_{cell}_block_{block}_Xte" \
                  .format(cell=cell_size, block=block_size)
print(
    "\tCreate " + path_to_data + output_filename + ".npy file ... ",
    end="",
    flush=True
)
np.save(path_to_data + output_filename, test_hog)
print("OK")


print("\nGradient histogram script completed in %0.2f seconds"
      % (time() - initial_time))
