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


def compute_hog_one_image(image, n_orientations, pixels_per_cell,
                          cells_per_block, method):
    image = image.astype('float')

    sy, sx = image.shape

    # Compute the horizontal gradient
    gx = np.zeros((sy, sx))
    gx[:,0] = image[:,1] - image[:,0]
    gx[:,-1] = image[:,-1] - image[:,-2]
    gx[:,1:-1] = 0.5 * (image[:,2:] - image[:,:-2])

    # Compute the vertical gradient
    gy = np.zeros((sy, sx))
    gy[0,:] = image[1,:] - image[0,:]
    gy[-1,:] = image[-1,:] - image[-2,:]
    gy[1:-1,:] = 0.5 * (image[2:,:] - image[:-2,:])

    cy, cx = pixels_per_cell
    by, bx = cells_per_block
    n_cellsx = int(sx // cx)
    n_cellsy = int(sy // cy)

    # Compute the gradient norm matrix and gradient orientations
    gradient_norm = np.sqrt(gx ** 2 + gy ** 2)
    orientations = (180 * (np.arctan2(gy, gx)) / np.pi) % 180

    orientation_histogram = np.zeros((n_cellsy, n_cellsx, n_orientations))
    for i in range(n_orientations):
        # Select correct orientations
        select_orientations = np.where(
            orientations < 180 / n_orientations * (i + 1),
            orientations,
            0
        )
        select_orientations = np.where(
            orientations >= 180 / n_orientations * i,
            select_orientations,
            0
        )

        # Select the gradient values according to the current orientation
        select_gradient_norm = np.where(
            select_orientations > 0,
            gradient_norm,
            0
        )

        # Compute the histogram of each cell for the current orientation
        for y in range(n_cellsy):
            for x in range(n_cellsx):
                hist_val = np.sum(
                    select_gradient_norm[cy*y:cy*(y+1),cx*x:cx*(x+1)]
                )
                hist_val = hist_val / (cx * cy)
                orientation_histogram[x,y,i] = hist_val

    # Normalize the blocks
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, n_orientations))
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5

            if method == 'L1':
                block_n = block / (np.sum(np.abs(block)) + eps)
            elif method == 'L1-sqrt':
                block_n = np.sqrt(block / (np.sum(np.abs(block)) + eps))
            elif method == 'L2':
                block_n = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
            elif method == 'L2-Hys':
                block_n = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
                block_n = np.minimum(block_n, 0.2)
                block_n = block_n / np.sqrt(np.sum(block ** 2) + eps ** 2)

            normalised_blocks[y, x, :] = block_n

    result_dimension = n_blocksy * n_blocksx * by * bx * n_orientations
    return normalised_blocks.reshape(result_dimension)


def hog_dimension(images_shape, n_orientations, pixels_per_cell, cells_per_block):
    sy, sx = images_shape
    cy, cx = pixels_per_cell
    by, bx = cells_per_block

    n_cellsx = int(sx // cx)
    n_cellsy = int(sy // cy)

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1

    return n_blocksy * n_blocksx * by * bx * n_orientations


def hog_filter(images, n_orientations, pixels_per_cell, cells_per_block,
               method):
    n_features = hog_dimension(
        images_shape=images[0].shape,
        n_orientations=n_orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    n_images = images.shape[0]

    percentage_filtered = 0
    percentage_step = 0.05

    hog_features = np.empty((n_images,n_features))
    for i in range(n_images):
        hog_features[i] = compute_hog_one_image(
            image=images[i],
            n_orientations=n_orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            method=method,
        )

        if i > (percentage_filtered + percentage_step) * n_images:
            percentage_filtered += percentage_step
            clean_percentage = int(round(100 * percentage_filtered,0))
            print("\t\t{percentage}% of the images have been filtered"
                  .format(percentage=clean_percentage))

    return hog_features
