from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QPlainTextEdit, QFileDialog, QScrollArea, QWidget
from PyQt5.QtCore import Qt
import sys
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import glob
import shutil
import subprocess
import json
from PIL import Image
from tqdm import tqdm
import yaml
import cv2
import numpy as np
from tqdm import tqdm

def txt_to_json(txt_file, img_file, output_dir, class_names):
    
    # Open the txt file
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    # Get the width and height of the image
    with Image.open(img_file) as img:
        width, height = img.size

    # Initialize an empty list to store the label data
    label_data = []

    # Process each line in the file
    for line in lines:
        # Split the line into components
        components = line.strip().split()

        # Extract the class id and the polygon points
        cls_id = int(components[0])
        points = [float(x) for x in components[1:]]

        # Group the points into (x, y) pairs
        points = list(zip(points[::2], points[1::2]))
        
        # Scale the points
        points = [(x * width, y * height) for x, y in points]
        
        # Create a dictionary for this label
        label_dict = {
            "label": class_names[cls_id],
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }

        # Add the label dictionary to the list
        label_data.append(label_dict)

    # Create a dictionary for the entire image
    img_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": label_data,
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": os.path.basename(img_file),
        "imageData": None
    }

    # Determine the name of the json file
    json_file = os.path.join(output_dir, os.path.basename(txt_file).replace('.txt', '.json'))

    # Write the json file
    with open(json_file, 'w') as f:
        json.dump(img_dict, f, ensure_ascii=False, indent=2)


def convert_all_txt_to_json(txt_dir, img_dir, output_dir, class_names):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif os.path.exists(output_dir):
        i = 1
        while os.path.exists(f"{output_dir}_{i}"):
            i += 1
        output_dir = f"{output_dir}_{i}"
        os.makedirs(output_dir)

    for txt_filename in tqdm(os.listdir(txt_dir)):
        if txt_filename.endswith('.txt'):
            # Construct the corresponding image filename and output filename
            base_filename = os.path.splitext(txt_filename)[0]
            output_filename = base_filename + '.json'

            if os.path.isfile(os.path.join(img_dir, base_filename + '.jpg')):
                img_filename = base_filename + '.jpg'
            elif os.path.isfile(os.path.join(img_dir, base_filename + '.png')):
                img_filename = base_filename + '.png'
            else:
                print(f"No corresponding .jpg or .png file found for {txt_filename}. Skipping.")
                continue

            # Convert the txt file to a json file
            txt_to_json(os.path.join(txt_dir, txt_filename), 
                        os.path.join(img_dir, img_filename),
                        output_dir,
                        class_names)
    print('Done.')
def get_class_names(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data.get('names', [])

def getLabelledFraction(json_path):

    # Load the JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load the image
    image_path = os.path.join(os.path.dirname(json_path), json_data['imagePath'])
    image = cv2.imread(image_path)
    
    overlay = np.zeros_like(image)  # Create a black image with the same dimensions as the original image
        

    # Iterate over the labels in the JSON data
    for shape in json_data['shapes']:
        # Get the class name and points for this label
        class_name = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        # Get the color for this class
        color = [1,1,1]

        # Create a polygon or rectangle and overlay it on the image
        if shape['shape_type'] == 'polygon':
            cv2.fillPoly(overlay, [points], color=color)
        elif shape['shape_type'] == 'rectangle':
            cv2.rectangle(overlay, tuple(points[0]), tuple(points[1]), color=color, thickness=-1)  # -1 thickness makes it filled

    
    # Count the number of zero-valued pixels
    total_pixels = overlay.shape[0] * overlay.shape[1]
    unlabelled_pixels = np.count_nonzero(np.all(overlay == 0, axis=2))
    # Calculate the fraction of unlabelled pixels
    fraction_labelled = 1-unlabelled_pixels / total_pixels
    
    return fraction_labelled
    
class MainApp(QApplication):
    def __init__(self, sys_argv):
        super(MainApp, self).__init__(sys_argv)
        
        # Main Layout
        self.layout = QVBoxLayout()

        # Parameters input
        parameters_layout = QHBoxLayout()
        self.layout.addLayout(parameters_layout)

        parameters_layout.addWidget(QLabel(text='Epochs:'))
        self.epochs_input = QLineEdit()
        self.epochs_input.setText("50")  # Set default value to 1000
        parameters_layout.addWidget(self.epochs_input)
        


        parameters_layout.addWidget(QLabel(text='Device:'))
        self.device_spinner = QComboBox()
        self.device_spinner.addItems(['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
        parameters_layout.addWidget(self.device_spinner)
        
        # Dropdown for model selection
        parameters_layout.addWidget(QLabel(text='Model:'))
        self.model_spinner = QComboBox()
        self.model_spinner.addItems(['yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt', 'yolov8p-seg.pt'])
        parameters_layout.addWidget(self.model_spinner)

        # Dropdown for task selection
        parameters_layout.addWidget(QLabel(text='Task:'))
        self.task_spinner = QComboBox()
        self.task_spinner.addItems(['segment'])
        parameters_layout.addWidget(self.task_spinner)

        # Dropdown for image size selection
        parameters_layout.addWidget(QLabel(text='Image Size:'))
        self.imgsz_spinner = QComboBox()
        self.imgsz_spinner.addItems(['640'])
        parameters_layout.addWidget(self.imgsz_spinner)

        # Dropdown for batch size selection
        parameters_layout.addWidget(QLabel(text='Batch Size:'))
        self.batch_spinner = QComboBox()
        self.batch_spinner.addItems(['4', '8', '16', '32'])
        parameters_layout.addWidget(self.batch_spinner)


        
   


        # Working Directory Selector
        self.working_dir_edit = QLineEdit()
        parameters_layout.addWidget(QLabel("Working Directory:"))
        parameters_layout.addWidget(self.working_dir_edit)
        self.select_working_dir_button = QPushButton("Select Working Directory")
        self.select_working_dir_button.clicked.connect(self.select_working_dir)
        parameters_layout.addWidget(self.select_working_dir_button)
        
        # Set the default working directory
        self.working_dir = "/home/triton/Desktop/SeaIce/data/journal-rgb-dataset/SemantixTemp"
        self.working_dir_edit.setText(self.working_dir)
        # Check if the directory exists and create it if necessary
        os.makedirs(self.working_dir, exist_ok=True)
        
        
        # Buttons
        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)

        load_button = QPushButton(text="Load imgs and labels")
        load_button.clicked.connect(self.load_images_labels)
        button_layout.addWidget(load_button)
        
        prepare_button = QPushButton(text="Prepare dataset in YOLO format")
        prepare_button.clicked.connect(self.prepare_dataset)
        button_layout.addWidget(prepare_button)

        train_button = QPushButton(text="Train data")
        train_button.clicked.connect(self.train_data)
        button_layout.addWidget(train_button)

        infer_button = QPushButton(text="Generate labels")
        infer_button.clicked.connect(self.generate_labels)
        button_layout.addWidget(infer_button)

        # Console
        self.results_console = QPlainTextEdit()
        self.results_console.setReadOnly(True)
        self.layout.addWidget(self.results_console)

        # Set Layout
        self.window = QWidget()
        self.window.setLayout(self.layout)
        self.window.show()

    def select_working_dir(self):
        self.working_dir = QFileDialog.getExistingDirectory(self.window, "Select Working Directory")
        self.working_dir_edit.setText(self.working_dir)


    def load_images_labels(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        filenames, _ = file_dialog.getOpenFileNames(self.window, filter="Images (*.jpg *.png)", options=options)
        
        
        # Create a list of image files that have a corresponding JSON file
        valid_files_img = [f for f in filenames if os.path.isfile(f.rsplit('.', 1)[0] + '.json')]
        self.results_console.appendPlainText(f"{len(valid_files_img)} files available, {len(filenames) - len(valid_files_img)} json files missing\n")

        # Create a corresponding list of JSON files
        valid_files_json = [f.rsplit('.', 1)[0] + '.json' for f in valid_files_img]
        
        # Filter the JSON files based on labelled area fraction
        labelled_area_fraction = 0.8

        valid_indices = []
        for i, f in tqdm(enumerate(valid_files_json), total=len(valid_files_json), desc="Processing files"):
            if getLabelledFraction(f) > labelled_area_fraction:
                valid_indices.append(i)

        # Update valid_files_img and valid_files_json to contain only images/JSONs with a sufficient labeled area fraction
        valid_files_img = [valid_files_img[i] for i in valid_indices]
        valid_files_json = [valid_files_json[i] for i in valid_indices]
        self.results_console.appendPlainText(f"Chosen {len(valid_files_img)}  files which have > {labelled_area_fraction} fraction of area labelled \n")



        
        
        imglabels_dir = os.path.join(self.working_dir, 'imglabels')
        os.makedirs(imglabels_dir, exist_ok=True)
        
        
        # Remove the directory if it exists, then recreate it
        if os.path.exists(imglabels_dir):
            shutil.rmtree(imglabels_dir)
        os.makedirs(imglabels_dir)
        
        json_file_paths = []

        for file_path in valid_files_img:
            # Copy the image file to the new directory
            shutil.copy(file_path, imglabels_dir)
            # Add the path of the corresponding json file
            json_file = file_path.rsplit('.', 1)[0] + '.json'
            json_file_paths.append(json_file)
            # Copy the json file to the new directory
            shutil.copy(json_file, imglabels_dir)


        self.results_console.appendPlainText("Copied files to: " + imglabels_dir)



    def prepare_dataset(self):
        self.results_console.appendPlainText("Prepare Dataset clicked\n")

        # Directories for input and output
        input_dir = os.path.join(self.working_dir, 'imglabels')
        #output_dir = os.path.join(self.working_dir, 'YOLOdataset')

        # Ensure output directory exists
        #os.makedirs(output_dir, exist_ok=True)

        # Get the path to the label list file
        label_list_file = os.path.join(self.working_dir, 'labels_file_list.txt')

        # Call labelme2yolo with subprocess
        subprocess.check_call(["labelme2yolo","--json_dir", input_dir ])

        self.results_console.appendPlainText("Dataset prepared in YOLO format\n")



    def train_data(self):
        self.results_console.appendPlainText("Train Data clicked\n")

        data_yaml = os.path.join(self.working_dir, 'imglabels', 'YOLODataset', 'dataset.yaml')
        cmd_args = [
            'yolo',
            f'task={self.task_spinner.currentText()}',
            'mode=train',
            f'epochs={self.epochs_input.text()}',
            f'data={data_yaml}',
            f'model={self.model_spinner.currentText()}',
            f'imgsz={self.imgsz_spinner.currentText()}',
            f'batch={self.batch_spinner.currentText()}',
            f'device={self.device_spinner.currentText()}',
            f'project=model_training',
            f'name=0'
        ]

        # Store the original directory
        original_dir = os.getcwd()

        # Change the working directory
        os.chdir(self.working_dir)

        # Call the subprocess
        subprocess.check_call(cmd_args)

        
        # Revert back to the original directory
        os.chdir(original_dir)

        self.results_console.appendPlainText("Training completed\n")



    def generate_labels(self):
        self.results_console.appendPlainText("Generate Labels clicked\n")

        # Select the image files
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        filenames, _ = file_dialog.getOpenFileNames(self.window, filter="Images (*.jpg *.png)", options=options)

        valid_files = [f for f in filenames]

        self.results_console.appendPlainText(f"{len(valid_files)} files loaded for prediction\n")

        # Prepare the directory for prediction
        imgs_dir = os.path.join(self.working_dir, 'imgs')
        os.makedirs(imgs_dir, exist_ok=True)

        # Remove the directory if it exists, then recreate it
        if os.path.exists(imgs_dir):
            shutil.rmtree(imgs_dir)
        os.makedirs(imgs_dir)

        for file_path in valid_files:
            # Copy the image file to the new directory
            shutil.copy(file_path, imgs_dir)

        self.results_console.appendPlainText("Copied files to: " + imgs_dir)

        # Perform the prediction
        cmd_args = [
            'yolo',
            'predict',
            f'model={os.path.join(self.working_dir, "model_training", "0", "weights", "best.pt")}',
            f'source={imgs_dir}',
            f'imgsz={self.imgsz_spinner.currentText()}',
            f'device={self.device_spinner.currentText()}',
            'save_txt=True',
            'save=False',
            'project=generated_labels',
            'name=0',
            'exist_ok=True'
        ]

        # Store the original directory
        original_dir = os.getcwd()

        # Change the working directory
        os.chdir(self.working_dir)

        # Call the subprocess
        subprocess.check_call(cmd_args)
        #process = subprocess.Popen(cmd_args)
        #process.wait()

        # After generating the labels, convert the txt files to json
        txt_dir = os.path.join(self.working_dir, "generated_labels", "0", "labels")
        img_dir = imgs_dir
        output_dir = os.path.join(self.working_dir, "jsonlabels")

        # Extract class names from dataset.yaml
        class_names = get_class_names(os.path.join(self.working_dir, 'imglabels', 'YOLODataset', 'dataset.yaml'))

        # Call convert_all_txt_to_json function
        convert_all_txt_to_json(txt_dir, img_dir, output_dir, class_names)
        self.results_console.appendPlainText("Converted TXT to JSON.\n")
        os.chdir(original_dir)


        self.results_console.appendPlainText("Label generation and conversion completed\n")





if __name__ == '__main__':
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
