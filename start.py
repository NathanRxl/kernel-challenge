import os


# Check that the data folder exists
if not os.path.isdir("data"):
    print("You first have to create the data directory and put the input " \
          "files in it")

# Check that the input files are in the data folder
input_filenames = ["Xte.csv", "Xtr.csv", "Ytr.csv"]
for file_name in input_filenames:
    if not os.path.exists("data/" + file_name):
        print("You first have to put the file {file_name} in the data directory"
              .format(file_name=file_name)
        )

# Run the preprocessing script
os.system("python3 preprocessing_script.py")

# The the Histogram of Oriented Gradients scripts
os.system("python3 hog_script.py 2 4")
os.system("python3 hog_script.py 4 4")
os.system("python3 hog_script.py 8 4")
os.system("python3 hog_script.py 16 1")

# Run the bilateral filtering script
os.system("python3 filter_script.py ")

# Run the pipeline to build the submission
os.system("python3 pipeline.py ")
