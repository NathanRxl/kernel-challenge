import pandas as pd
from datetime import datetime


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


def create_submission(y_pred, submission_folder_path="submissions/",
                      output_filename=None):
    """
    Create a submit csv file from mids_prediction.
    mids_prediction should be of the form : {mid: list_of_recipients}
    """
    if output_filename is None:
        output_filename = (
            "submission_" +
            datetime.now().strftime("%Y_%m_%d_%M_%S") +
            ".txt"
        )

    submission = pd.read_csv("submissions/sample_submission.csv")
    submission["Prediction"] = y_pred
    submission.to_csv(
        submission_folder_path + output_filename,
        index=False
    )