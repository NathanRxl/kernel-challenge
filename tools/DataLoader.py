import pandas as pd


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
            return X_train, y_train
        else:
            X_test = pd.read_csv(self.path_to_data + X_data_filename)
            return X_test
