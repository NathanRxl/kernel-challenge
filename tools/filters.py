import numpy as np

from cv2 import GaussianBlur
from skimage.restoration import denoise_bilateral


def gaussian_filter(images, window_size, sigma, color=False):
    """
    Apply a gaussian filter to every image in images.
    images should be of shape : (n_images, 32, 32, 3) or (n_images, 32, 32)
    """
    n_images = images.shape[0]
    window = (window_size, window_size)
    filtered_images = np.zeros(images.shape)
    for i in range(n_images):
        filtered_images[i] = GaussianBlur(images[i], window, sigma)

    if color:
        images_size = 32 * 32 * 3
    else:
        images_size = 32 * 32

    return filtered_images.reshape((n_images, images_size))


def skimage_bilateral_filter(images, d, sigma, color=False):
    """
    Apply a bilateral filter to every image in images.
    images should be of shape : (n_images, 32, 32, 3) or (n_images, 32, 32)
    """
    n_images = images.shape[0]
    filtered_images = np.zeros(images.shape)
    for i in range(n_images):
        # 0.47 is because denoise_bilateral only accepts positive values
        filtered_images[i] = denoise_bilateral(images[i] + 0.47, d, sigma, sigma)

    if color:
        images_size = 32 * 32 * 3
    else:
        images_size = 32 * 32

    return filtered_images.reshape((n_images, images_size))


def bilateral_filter_one_image(image, window_size, sigma):
    """
    Apply a bilateral filter to one color image.
    image should be of shape : (32, 32, 3) or (n_images, 32, 32)
    """
    (height, width, dim) = image.shape
    filtered_image = np.empty(image.shape)

    window_range = range(- window_size, window_size + 1)
    (X, Y) = np.meshgrid(window_range, window_range)
    distance_weights = np.exp(- (np.square(X) + np.square(Y)) / (2 * sigma ** 2))

    for x in range(height):
        for y in range(width):
            x_low_bound = max(x - window_size, 0)
            x_up_bound = min(x + window_size, 31)
            y_low_bound = max(y - window_size, 0)
            y_up_bound = min(y + window_size, 31)
            I = image[x_low_bound:x_up_bound, y_low_bound:y_up_bound, :]

            color_distance = np.sum(np.square(I - image[x, y, :]))
            color_weight = np.exp(- color_distance / (2 * sigma ** 2))

            min_x = x_low_bound - x + window_size
            max_x = x_up_bound - x + window_size
            min_y = y_low_bound - y + window_size
            max_y = y_up_bound - y + window_size

            global_weight = (
                color_weight * distance_weights[min_x:max_x, min_y:max_y]
            )
            weight_sum = np.sum(global_weight)

            filtered_image[x, y, 0] = np.sum(global_weight*I[:, :, 0]) / weight_sum
            filtered_image[x, y, 1] = np.sum(global_weight*I[:, :, 1]) / weight_sum
            filtered_image[x, y, 2] = np.sum(global_weight*I[:, :, 2]) / weight_sum

    return filtered_image


def bilateral_filter(images, window_size, sigma):
    """
    Apply a bilateral filter to every image in images.
    images should be of shape : (n_images, 32, 32, 3) or (n_images, 32, 32)
    window_size is a positive integer and sigma is a positive float.
    """
    (n_images, height, width, dim) = images.shape
    filtered_images = np.empty(images.shape)

    percentage_filtered = 0
    percentage_step = 0.05

    print("\n\t\tStart applying a bilateral filter ...\n")

    for i in range(n_images):
        filtered_images[i] = bilateral_filter_one_image(
            image=images[i],
            window_size=window_size,
            sigma=sigma,
        )

        if i > (percentage_filtered + percentage_step) * n_images:
            percentage_filtered += percentage_step
            clean_percentage = int(round(100 * percentage_filtered,0))
            print("\t\t{percentage}% of the images have been filtered"
                  .format(percentage=clean_percentage))

    return filtered_images.reshape((n_images, 32 * 32 * 3))
