from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QPlainTextEdit, QFileDialog, QScrollArea, QWidget
from PyQt5.QtCore import Qt
import sys
import os
import glob



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

    def load_images_labels(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        filenames, _ = file_dialog.getOpenFileNames(filter="Images (*.jpg *.png)", options=options)

        valid_files = [f for f in filenames if os.path.isfile(f.rsplit('.', 1)[0] + '.json')]

        self.results_console.appendPlainText(f"{len(valid_files)} files loaded, {len(filenames) - len(valid_files)} json files missing\n")


    def prepare_dataset(self):
        # Implement logic here
        self.results_console.appendPlainText("Prepare Dataset clicked\n")

    def train_data(self):
        # Implement logic here
        self.results_console.appendPlainText("Train Data clicked\n")

    def generate_labels(self):
        # Implement logic here
        self.results_console.appendPlainText("Generate Labels clicked\n")


if __name__ == '__main__':
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
