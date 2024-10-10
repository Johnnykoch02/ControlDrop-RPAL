# label_app.py
import os
os.environ["SIM_GUI"] = "true"
import sys
import threading
import time
from enum import Enum

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QInputDialog,
    QComboBox,
)
from PyQt5.QtCore import Qt

from control_dropping_rpal.RL.control_dropping_env import (
    GraspShape, GraspFunction, SceneDifficulty, SceneManager, BerrettHandGym
)

class LabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize SceneManager and BerrettHandGym
        self.scene_manager = SceneManager(detailed_training=True)
        self._env_config = dict(
            detailed_training=True,
            object_quantity=7,
            difficulties=[SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD],
            is_val=True,
        )
        self.berret_hand_gym = BerrettHandGym(**self._env_config)

        self.current_index = 0
        self.total_size = self.scene_manager.dataset_size()

        self.enum_classes = {
            'SceneDifficulty': SceneDifficulty,
            'GraspShape': GraspShape,
            'GraspFunction': GraspFunction
        }

        self.setup_ui()

        self.is_setting_scene = False
        self.last_index_change_time = 0

        # Load the scene at the initial index
        self.load_scene(self.current_index)

    def setup_ui(self):
        self.setWindowTitle("Scene Metadata Labeling App")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Index Display and Controls
        index_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_index)
        index_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_index)
        index_layout.addWidget(self.next_button)

        self.index_label = QLabel("")
        index_layout.addWidget(self.index_label)

        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("Enter index")
        index_layout.addWidget(self.index_input)

        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.go_to_index)
        index_layout.addWidget(self.go_button)

        main_layout.addLayout(index_layout)

        # Metadata Display
        self.metadata_widgets = {}  # key: (label_widget, value_widget)

        self.metadata_layout = QVBoxLayout()

        # Scroll area for metadata fields
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        self.scroll_layout = scroll_layout

        main_layout.addWidget(scroll_area)

        # Add New Metadata Button
        self.add_button = QPushButton("Add Metadata")
        self.add_button.clicked.connect(self.add_metadata_field)
        main_layout.addWidget(self.add_button)

        # Save Button
        self.save_button = QPushButton("Save Metadata")
        self.save_button.clicked.connect(self.save_metadata)
        main_layout.addWidget(self.save_button)

        self.update_index_label()

    def update_index_label(self):
        self.index_label.setText(f"Index: {self.current_index} / {self.total_size - 1}")

    def load_scene(self, index):
        if self.is_setting_scene:
            return

        # Implement 3-second delay before setting scene again
        current_time = time.time()
        if current_time - self.last_index_change_time < 3:
            QMessageBox.warning(
                self, "Wait", "Please wait 3 seconds before changing index again."
            )
            return

        self.last_index_change_time = current_time

        self.is_setting_scene = True

        # Clear metadata display
        for widgets in self.metadata_widgets.values():
            for widget in widgets:
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
        self.metadata_widgets.clear()

        # Load scene and metadata
        scene, metadata = self.scene_manager.get_data_index(index)
        self.current_metadata = metadata.copy()

        # Display metadata
        for key, value in metadata.items():
            if key != "index":  # Skip 'index' key
                self.add_metadata_row(key, value)

        self.update_index_label()

        # Start a thread to set up the scene
        threading.Thread(target=self.setup_scene_thread, args=(scene,)).start()

    def setup_scene_thread(self, scene):
        success = self.berret_hand_gym.set_sim_scene(scene)
        if success:
            print("Scene set up successfully.")
        else:
            print("Failed to set up scene.")
        self.is_setting_scene = False

    def add_metadata_row(self, key, value):
        h_layout = QHBoxLayout()

        label = QLabel(key)
        h_layout.addWidget(label)

        if isinstance(value, Enum):
            value_input = QComboBox()
            enum_class = type(value)
            for enum_value in enum_class:
                value_input.addItem(enum_value.name, enum_value)
            value_input.setCurrentText(value.name)
        else:
            value_input = QLineEdit()
            value_input.setText(str(value))

        h_layout.addWidget(value_input)

        # Store widgets
        self.metadata_widgets[key] = (label, value_input)
        self.scroll_layout.addLayout(h_layout)

    def add_metadata_field(self):
        # Get existing keys and their types
        existing_keys = list(self.current_metadata.keys())
        existing_types = [type(v).__name__ for v in self.current_metadata.values()]

        # Open dialog to select existing key or enter new one
        key, ok = QInputDialog.getItem(
            self, "Select or Enter Key", "Key:", 
            existing_keys + ["<New Key>"], editable=True
        )
        if not ok or not key:
            return

        if key == "<New Key>":
            key, ok = QInputDialog.getText(self, "Input", "Enter new metadata key:")
            if not ok or not key:
                return

        # Determine data type
        if key in self.current_metadata:
            data_type = type(self.current_metadata[key]).__name__
        else:
            data_types = list(set(existing_types + list(self.enum_classes.keys()) + ["str", "int", "float"]))
            data_type, ok = QInputDialog.getItem(
                self, "Select Data Type", "Data Type:", data_types, editable=False
            )
            if not ok:
                return

        # Get value based on data type
        if data_type in self.enum_classes:
            enum_class = self.enum_classes[data_type]
            enum_names = [e.name for e in enum_class]
            value_name, ok = QInputDialog.getItem(
                self, "Select Value", f"Select {data_type} value:", enum_names, editable=False
            )
            if not ok:
                return
            value = enum_class[value_name]
        else:
            value, ok = QInputDialog.getText(self, "Input", "Enter value:")
            if not ok:
                return
            # Convert value to the correct type
            try:
                if data_type == "int":
                    value = int(value)
                elif data_type == "float":
                    value = float(value)
                # For str, no conversion needed
            except ValueError:
                QMessageBox.warning(self, "Error", f"Invalid value for type {data_type}")
                return

        # Add to current metadata
        self.current_metadata[key] = value
        self.add_metadata_row(key, value)

    def save_metadata(self):
        metadata = {}
        for key, widgets in self.metadata_widgets.items():
            value_widget = widgets[1]
            if isinstance(value_widget, QComboBox):
                value = value_widget.currentData()
            else:
                value_str = value_widget.text()
                original_value = self.current_metadata.get(key)
                if isinstance(original_value, int):
                    value = int(value_str)
                elif isinstance(original_value, float):
                    value = float(value_str)
                elif isinstance(original_value, Enum):
                    enum_class = type(original_value)
                    value = enum_class[value_str]
                else:
                    value = value_str
            metadata[key] = value

        # Save metadata using SceneManager
        self.scene_manager.set_metadata_at_index(self.current_index, metadata)
        QMessageBox.information(self, "Saved", "Metadata saved successfully.")

    def prev_index(self):
        self.current_index = (self.current_index - 1) % self.total_size
        self.load_scene(self.current_index)

    def next_index(self):
        self.current_index = (self.current_index + 1) % self.total_size
        self.load_scene(self.current_index)

    def go_to_index(self):
        try:
            index = int(self.index_input.text())
            if 0 <= index < self.total_size:
                self.current_index = index
                self.load_scene(self.current_index)
            else:
                QMessageBox.warning(self, "Error", "Index out of range.")
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid index.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())