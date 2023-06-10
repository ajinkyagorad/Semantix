from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QPlainTextEdit, QFileDialog, QScrollArea, QWidget
from PyQt5.QtCore import Qt
import sys
import os
import glob
import shutil
import subprocess
import json

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
        parameters_layout.addWidget(self.epochs_input)
        
        parameters_layout.addWidget(QLabel(text='Learning Rate:'))
        self.lr_input = QLineEdit()
        parameters_layout.addWidget(self.lr_input)

        parameters_layout.addWidget(QLabel(text='Device:'))
        self.device_spinner = QComboBox()
        self.device_spinner.addItems(['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
        parameters_layout.addWidget(self.device_spinner)
        
   


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

        valid_files = [f for f in filenames if os.path.isfile(f.rsplit('.', 1)[0] + '.json')]

        self.results_console.appendPlainText(f"{len(valid_files)} files loaded, {len(filenames) - len(valid_files)} json files missing\n")

        imglabels_dir = os.path.join(self.working_dir, 'imglabels')
        os.makedirs(imglabels_dir, exist_ok=True)

        json_file_paths = []

        for file_path in valid_files:
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
        # Implement logic here
        self.results_console.appendPlainText("Train Data clicked\n")

    def generate_labels(self):
        # Implement logic here
        self.results_console.appendPlainText("Generate Labels clicked\n")


if __name__ == '__main__':
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
