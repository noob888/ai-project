# ai-project

# Project Overview
This project aims to distinguish between real and fake faces using the following methods:

DeepFace Verification: Verifies if two images contain the same face.
Haar Cascade Classifiers: Detects faces and eyes in images to analyze the presence of open or closed eyes.

$ The workflow involves:
Displaying images to verify their content.
Using DeepFace for face comparison.
Detecting faces and eyes using Haar cascades.
Drawing bounding boxes around detected features and determining eye status.

# Installation
Ensure you have Python installed. Then, install the required packages using pip:
```
pip install deepface opencv-python matplotlib
```

You also need to download Haar Cascade files for face and eye detection:
Haar Cascade Face Classifier (easily available on Google )
Haar Cascade Eye Classifier
Save these files to a directory on your machine and update the paths in the script.

# Usage
Prepare Image Paths:
Update the paths in the script to point to your test images:
```
img1_path = '{ADD_PATH_TO_FACE}'
img2_path = '{ADD_PATH_TO_FACE_WITH_OPEN_EYES}'
img3_path = '{ADD_PATH_TO_FACE_WITH_CLOSED_EYES}'
```

# Run the Script:
Execute the Python script to analyze the images:
```
python ai-model.py
```

# Review Results:
The script will display images with detected faces and eyes.
It will print whether the face is real or fake based on the presence of open or closed eyes.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This is just a development model, project is not licensed. Production model is licensed and not available.
