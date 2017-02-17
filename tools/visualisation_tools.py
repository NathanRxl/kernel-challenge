"""
These tools are mainly designed to be used in a Jupyter notebook.
However they can be used in script as well.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_image(X_train, y_train, image_nb, color=False):
    if color:
        digit_image = np.zeros(shape=(32, 32, 3))
        # Red
        digit_image[:, :, 0] = (
            X_train.loc[image_nb][:1024].values.reshape(32, 32, order='F')
        )
        # Green
        digit_image[:, :, 1] = (
            X_train.loc[image_nb][1024:2048].values.reshape(32, 32, order='F')
        )
        # Blue
        digit_image[:, :, 2] = (
            X_train.loc[image_nb][2048:].values.reshape(32, 32, order='F')
        )
    else:
        # Grey
        digit_image = X_train.loc[image_nb].values.reshape(32, 32, order='F')
    plt.imshow(digit_image, cmap='gray')
    plt.title("Image of a {}".format(y_train[image_nb]))
    plt.show()
