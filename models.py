import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from penalizations import ridge, grad_ridge


class KernelLogisticRegression:
    """
    Logistic Regression model
    """

    def __init__(self, lbda=0, kernel=None, penalty=ridge,
                 grad_penalty=grad_ridge, multi_class=None):
        """
        lbda is the regularisation parameter
        penalty is a callable with signature penalty(lbda, f_x)
        coefs_ is such that the predictor f(x_i) = <self.coefs_, x_i>
        """
        self.lbda = lbda
        self.kernel = kernel
        self.penalty = penalty
        self.grad_penalty = grad_penalty
        self.alpha = None
        self.w = None
        self.step = 1
        self.n_samples = None
        self.n_features = None
        self.multi_class = multi_class
        self.n_classes = None

    @staticmethod
    def _sigmoid(x):
        """ Sigmoid function (overflow-proof) """
        positive_idx = x > 0
        sigmoid_result = np.empty(x.size)
        sigmoid_result[positive_idx] = 1. / (1 + np.exp(-x[positive_idx]))
        exp_x_negative = np.exp(x[~positive_idx])
        sigmoid_result[~positive_idx] = exp_x_negative / (1. + exp_x_negative)
        return sigmoid_result

    def _multinomial(self, x):
        """ Multinomial function (overflow-proof) """
        exp_x = np.exp(x - np.max(x, axis=1).reshape((len(x),1)))
        sum_exp = np.sum(exp_x, axis=1).reshape(len(exp_x), 1)
        multinomial_result = exp_x / sum_exp
        return multinomial_result

    def _which_size(self, X_or_K):
        if self.kernel is None:
            X = X_or_K
            return X.shape[0], X.shape[1]
        elif self.kernel == "precomputed":
            K = X_or_K
            return len(K)
        else:
            raise (
                ValueError(
                    "kernel argument {} "
                    "unsupported by LogisticRegression"
                    .format(self.kernel))
            )

    @staticmethod
    def _which_n_classes(y):
        return len(np.unique(y))

    def _penalized_logloss(self, w_or_alpha, X_or_K, y):
        """
        Compute the logistic loss
        given X and y according to the current self.coefs_
        """

        if self.kernel is None:
            w = w_or_alpha
            X = X_or_K
            if self.multi_class == "multinomial" and self.n_classes > 2:
                w_reshape = w.reshape(self.n_features, self.n_classes)
                f_x = X.dot(w_reshape)

                max_f_x = np.max(f_x, axis=1).reshape(self.n_samples, 1)
                log_sum_exp = (
                    max_f_x + np.log(np.sum(np.exp(f_x - max_f_x), axis=1))
                )
                y_mask = (
                    y.copy().reshape(self.n_samples, 1)
                    == np.arange(self.n_classes)
                )
                logloss_result = - np.sum(
                    (1 / self.n_samples) * (f_x[y_mask] - log_sum_exp)
                )

                penalty_result = self.penalty(self.lbda, w_reshape)

            elif self.multi_class is None:
                f_x = X.dot(w)
                logloss_result = (
                    (1 / self.n_samples) * np.sum(np.log(1 + np.exp(- y * f_x)))
                )

                penalty_result = self.penalty(self.lbda, w)
            else:
                raise (
                    ValueError(
                        "multi_class argument {} "
                        "unsupported by LogisticRegression"
                        .format(self.multi_class))
                )

        elif self.kernel == "precomputed":
            alpha = w_or_alpha
            K = X_or_K
            if self.multi_class == "multinomial" and self.n_classes > 2:
                alpha_reshape = alpha.reshape(self.n_samples, self.n_classes)
                f_x = K.dot(alpha_reshape)

                max_f_x = np.max(f_x, axis=1).reshape(self.n_samples, 1)
                log_sum_exp = (
                    max_f_x + np.log(np.sum(np.exp(f_x - max_f_x), axis=1))
                )
                y_mask = (
                    y.copy().reshape(self.n_samples, 1)
                    == np.arange(self.n_classes)
                )
                logloss_result = - np.sum(
                    (1 / self.n_samples) * (f_x[y_mask] - log_sum_exp)
                )

                penalty_result = self.penalty(self.lbda, alpha_reshape, K)

            elif self.multi_class is None:
                f_x = K.dot(alpha)
                logloss_result = (
                    (1 / self.n_samples) * np.sum(
                        np.log(1 + np.exp(- y * f_x)))
                )

                penalty_result = self.penalty(self.lbda, alpha, K)
            else:
                raise (
                    ValueError(
                        "multi_class argument {} "
                        "unsupported by LogisticRegression"
                        .format(self.multi_class))
                )

        else:
            raise (
                ValueError(
                    "kernel argument {} "
                    "unsupported by LogisticRegression"
                    .format(self.kernel))
            )

        return logloss_result + penalty_result

    def _grad_penalized_logloss(self, w_or_alpha, X_or_K, y):
        """
        Compute the logloss gradient
        given X and y according to the current self.coefs_
        """

        if self.kernel is None:
            w = w_or_alpha
            X = X_or_K
            if self.multi_class == "multinomial" and self.n_classes > 2:
                w_reshape = w.reshape(self.n_features, self.n_classes)
                f_x = X.dot(w_reshape)

                max_f_x = np.max(f_x, axis=1).reshape(self.n_samples, 1)
                log_sum_exp = (
                    max_f_x[:, 0] + np.log(np.sum(np.exp(f_x - max_f_x), axis=1))
                ).reshape(self.n_samples, 1)

                y_mask = (
                    y.copy().reshape(self.n_samples, 1)
                    == np.arange(self.n_classes)
                )

                logloss_grad_result = (
                    (1 / self.n_samples)
                    * np.dot(X.T, np.exp(f_x) / log_sum_exp - y_mask)
                ).reshape((self.n_features * self.n_classes, ))

                grad_penalty_result = (
                    self.grad_penalty(self.lbda, w_reshape).reshape(
                        (self.n_features * self.n_classes, )
                    )
                )
            else:
                f_x = X.dot(w)
                exp_margin = np.exp(y * f_x).reshape((self.n_samples, 1))
                labels = y.copy().reshape((self.n_samples, 1))
                logloss_grad_result = (
                    - (1 / self.n_samples)
                    * np.sum(labels * X / (1 + exp_margin), axis=0)
                )

                grad_penalty_result = self.grad_penalty(self.lbda, w)

        elif self.kernel == "precomputed":
            alpha = w_or_alpha
            K = X_or_K
            if self.multi_class == "multinomial" and self.n_classes > 2:
                alpha_reshape = alpha.reshape(self.n_samples, self.n_classes)
                f_x = K.dot(alpha_reshape)

                max_f_x = np.max(f_x, axis=1).reshape(self.n_samples, 1)
                log_sum_exp = (
                    max_f_x[:, 0] + np.log(np.sum(np.exp(f_x - max_f_x), axis=1))
                ).reshape(self.n_samples, 1)

                y_mask = (
                    y.copy().reshape(self.n_samples, 1)
                    == np.arange(self.n_classes)
                )

                logloss_grad_result = (
                    (1 / self.n_samples)
                    * np.dot(K.T, np.exp(f_x) / log_sum_exp - y_mask)
                ).reshape((self.n_samples * self.n_classes, ))

                grad_penalty_result = (
                    self.grad_penalty(self.lbda, alpha_reshape, K).reshape(
                        (self.n_samples * self.n_classes, )
                    )
                )
            else:
                f_x = K.dot(alpha)
                exp_margin = np.exp(y * f_x).reshape((self.n_samples, 1))
                labels = y.copy().reshape((self.n_samples, 1))
                logloss_grad_result = (
                    - (1 / self.n_samples)
                    * np.sum(labels * K / (1 + exp_margin), axis=0)
                )
                grad_penalty_result = self.grad_penalty(self.lbda, alpha, K)
        else:
            raise (
                ValueError(
                    "kernel argument {} "
                    "unsupported by LogisticRegression"
                    .format(self.kernel))
            )

        return logloss_grad_result + grad_penalty_result

    def fit(self, X_or_K, y):
        """
            Train model on X and y
        """
        self.n_classes = self._which_n_classes(y)

        if self.kernel is None:
            X = X_or_K
            self.n_samples, self.n_features = self._which_size(X)

            if self.multi_class == "multinomial":
                # fmin_l_bfgs_b only optimizes 1 dimensional array's
                w0 = np.zeros((self.n_features * self.n_classes, ))
            elif self.multi_class is None:
                w0 = np.zeros((self.n_features, ))
            else:
                raise (
                    ValueError(
                        "multi_class argument {} unsupported by "
                        "LogisticRegression"
                        .format(self.multi_class))
                )
            self.w, logloss_at_minimum, information = (
                fmin_l_bfgs_b(
                    args=(X, y),
                    func=self._penalized_logloss,
                    x0=w0,
                    fprime=self._grad_penalized_logloss
                )
            )
        elif self.kernel == "precomputed":
            K = X_or_K
            self.n_samples = self._which_size(K)

            if self.multi_class == "multinomial":
                # fmin_l_bfgs_b only optimizes 1 dimensional array's
                alpha0 = np.zeros((self.n_samples * self.n_classes, ))
            elif self.multi_class is None:
                alpha0 = np.zeros((self.n_samples, ))
            else:
                raise (
                    ValueError(
                        "multi_class argument {} unsupported by "
                        "LogisticRegression"
                        .format(self.multi_class))
                )

            self.alpha, logloss_at_minimum, information = (
                fmin_l_bfgs_b(
                    args=(K, y),
                    func=self._penalized_logloss,
                    x0=alpha0,
                    fprime=self._grad_penalized_logloss
                )
            )
        else:
            raise (
                ValueError(
                    "kernel argument {} "
                    "unsupported by LogisticRegression"
                    .format(self.kernel))
            )

        if information['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            warnings.warn(
                "In LogisticRegression, optimisation finished on an "
                "abnormal termination in linesearch. This could lead to "
                "non optimal solution and inefficient computations.",
                RuntimeWarning
            )

        if self.multi_class == "multinomial" and self.n_classes > 2:
            if self.kernel is None:
                self.w = self.w.reshape(
                    (self.n_features, self.n_classes)
                )
            elif self.kernel == "precomputed":
                self.alpha = self.alpha.reshape(
                    (self.n_samples, self.n_classes)
                )
            else:
                raise (
                    ValueError(
                        "kernel argument {} "
                        "unsupported by LogisticRegression"
                        .format(self.kernel))
                )

    def predict_proba(self, X_or_K):
        """
            Compute probability of X class base on self.coefs_
        """
        if self.multi_class == 'multinomial' and self.n_classes > 2:
            if self.kernel is None:
                X = X_or_K
                y_pred_proba = self._multinomial(np.dot(X, self.w))
                return y_pred_proba
            elif self.kernel == "precomputed":
                K = X_or_K
                y_pred_proba = self._multinomial(
                    np.dot(K, self.alpha)
                )
                return y_pred_proba
            else:
                raise (
                    ValueError(
                        "kernel argument {} "
                        "unsupported by LogisticRegression"
                        .format(self.kernel))
                )
        elif self.multi_class is None:
            if self.kernel is None:
                X = X_or_K
                y_pred_proba = self._sigmoid(np.dot(X, self.w))
                return np.column_stack((1 - y_pred_proba, y_pred_proba))
            elif self.kernel == "precomputed":
                K = X_or_K
                y_pred_proba = self._sigmoid(np.dot(K, self.alpha))
                return np.column_stack((1 - y_pred_proba, y_pred_proba))
            else:
                raise (
                    ValueError(
                        "kernel argument {} "
                        "unsupported by LogisticRegression"
                        .format(self.kernel))
                )

    def predict(self, X_or_K):
        return np.argmax(self.predict_proba(X_or_K), axis=1)
