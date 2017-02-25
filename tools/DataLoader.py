import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, path_to_data="data/"):
        self.path_to_data = path_to_data

    def load_labels_only(self, y_data_filename):
        y_train = (
            pd.read_csv(
                self.path_to_data + y_data_filename,
                index_col=None,
                usecols=["Prediction"]
            )
            .values.reshape((5000,))
        )
        return y_train

    def load_data(self, X_data_filename, y_data_filename=None):
        if y_data_filename is not None:
            X_train = pd.read_csv(self.path_to_data + X_data_filename)
            y_train = self.load_labels_only(y_data_filename)
            return X_train.as_matrix(), y_train
        else:
            X_test = pd.read_csv(self.path_to_data + X_data_filename)
            return X_test.as_matrix()

    def load_kernel(self, K_data_filename, y_data_filename=None,
                    train_samples=None):
        if y_data_filename is not None:
            K_train = np.fromfile(self.path_to_data + K_data_filename)
            n_samples = int(np.sqrt(len(K_train)))
            K_train = K_train.reshape(n_samples, n_samples)
            y_train = self.load_labels_only(y_data_filename)
            return K_train, y_train
        else:
            K_test = np.fromfile(self.path_to_data + K_data_filename)
            K_test = K_test.reshape(
                (int(len(K_test) / train_samples), train_samples)
            )
            return K_test
