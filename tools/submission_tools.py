import pandas as pd
from datetime import datetime


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
