import numpy as np
import pandas as pd


def rgb2grey(X):
    grey_X = X.copy()
    for red_pixel_nb in range(1024):
        green_pixel_nb = red_pixel_nb + 1024
        blue_pixel_nb = green_pixel_nb + 1024
        grey_X["grey_" + str(red_pixel_nb)] = (
            0.2989 * grey_X[red_pixel_nb] +
            0.5870 * grey_X[green_pixel_nb] +
            0.1140 * grey_X[blue_pixel_nb]
        )
    return grey_X.filter(regex="grey_*")


def color_images_from_df(color_df):
    """
    Convert the dataframe of the color images into a numpy matrix of size :
    (number_images, 32, 32, 3)
    """
    X = color_df.as_matrix()
    n_images = X.shape[0]
    color_images = np.zeros((n_images, 32, 32, 3), dtype="float32")
    for i in range(3):
        color_images[:, :, :, i] = (X[:, 32*32*i:32*32*(i + 1)]
                                    .reshape(n_images, 32, 32))
    return color_images


def gray_images_from_df(gray_df):
    """
    Convert the dataframe of the gray images into a numpy matrix of shape :
    (number_images, 32, 32)
    """
    images = gray_df.as_matrix().astype(np.float32)
    n_images = images.shape[0]
    return images.reshape((n_images, 32, 32))


def histogram_df(X_df, bins):
    global_histogram = np.histogram(X_df, bins=bins, normed=True)
    histogram_range = global_histogram[1].min(), global_histogram[1].max()

    hist_dict = dict()

    for row_idx in range(len(X_df)):
        hist_dict[row_idx] = np.histogram(
            X_df.loc[row_idx],
            bins=bins,
            range=histogram_range,
            normed=False
        )[0]

    return pd.DataFrame(data=hist_dict).T / X_df.shape[1]
