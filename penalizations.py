import numpy as np


def lasso(lbda, w):
    """
        Value of the Lasso penalization at coefs_ with strength lbda
    """
    return lbda * np.linalg.norm(w, 1)


def grad_lasso(lbda, w):
    """
        Value of the gradient of the lasso penalization
        in coefs_ with strength lbda
    """
    grad_result = np.zeros_like(w)
    grad_result[w < 0] = - lbda
    grad_result[w > 0] = lbda
    return grad_result


def prox_lasso(lbda, w):
    """
        Proximal operator for lbda times the Lasso at coefs_
        Soft thresholding
    """
    return np.sign(w) * np.maximum(np.absolute(w) - lbda, 0)


def kernel_ridge(lbda, alpha, K):
    return lbda * np.sum(alpha.T.dot(K).dot(alpha)) / 2


def grad_kernel_ridge(lbda, alpha, K):
    return lbda * K.dot(alpha)


def ridge(lbda, w):
    """
        Value of the ridge penalization at coefs_ with strength lbda
    """
    return lbda * (np.linalg.norm(w, 2) ** 2) / 2


def grad_ridge(lbda, w):
    """
        Value of the gradient of the ridge penalization
        in coefs_ with strength lbda
    """
    return lbda * w


def prox_ridge(lbda, w):
    """
        Ridge proximal operator at coefs_ with strength lbda
        Shrinkage operator
    """
    return w / (1 + lbda)
