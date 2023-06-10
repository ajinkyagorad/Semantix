# Semantix: An Open Source YOLO-based Image Labeling Tool

Welcome to **Semantix**, a GUI-based open-source tool designed to simplify the process of labeling and training data for object detection using the YOLO (You Only Look Once) algorithm. Semantix streamlines the workflow of preparing a dataset, training the model, and generating predictions.

## Key Features (in progress)
- Easy-to-use, user-friendly GUI.
- Supports .jpg and .png image files.
- Allows users to load images and corresponding labels with a single click.
- Facilitates dataset preparation in YOLO format.
- Provides options to set key training parameters like epochs and learning rate.
- Integrated support for multiple CUDA devices.
- Displays real-time training logs in the console.
- Automatic saving of weight files in the YOLO file system upon completion of training.
- Enables label generation for a large set of images.
- Supports Semantic Segmentation output with polygons (or rectangles) for different object classes.

## Installation
Clone this repository and navigate into it:
```bash
git clone https://github.com/yourgithubusername/Semantix.git
cd Semantix
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python trainlabelGUI.py
```

## License

This project is licensed under the terms of the MIT License. For more details, see the LICENSE file.