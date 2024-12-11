import tkinter as tk
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class ObjectDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TensorFlow Object Detection Inference Panel")
        self.geometry('1024x768')
        
        # Preload model and category index
        self.category_index = self._load_category_index()
        self.model = self._load_detection_model()
        
        self._create_ui()

    @staticmethod
    def _load_category_index():
        """Efficiently load COCO category index."""
        return {
            1: {'id': 1, 'name': 'person'},
            # ... (rest of the original category index)
            # Consider loading from a JSON file for larger datasets
        }

    @staticmethod
    def _load_detection_model():
        """Optimize model loading with error handling."""
        try:
            # Clear any existing TensorFlow sessions
            tf.keras.backend.clear_session()
            
            # Use eager execution for better performance
            tf.config.run_functions_eagerly(False)
            
            # Load model with performance optimization
            model = tf.saved_model.load(
                "centernet_resnet50_v2_512x512_coco17_tpu-8/saved_model", 
                options=tf.saved_model.LoadOptions(experimental_skip_checkpoint=True)
            )
            return model
        except Exception as e:
            print(f"Model loading error: {e}")
            return None

    def _create_ui(self):
        """Create the application user interface."""
        # Main container
        main_container = tk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left frame for image and detection
        left_frame = tk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right frame for object list
        right_frame = tk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Create menu
        self._create_menu()

        # Perform object detection
        self._perform_object_detection(left_frame, right_frame)

    def _create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self._open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _perform_object_detection(self, left_frame, right_frame):
        """Centralized method for object detection and visualization."""
        # Default image path
        file_path = "./dog_bike_car.jpg"
        
        try:
            # Load and preprocess image
            image_np = self._load_image_to_numpy(file_path)
            input_tensor = np.expand_dims(image_np, 0)

            # Perform inference
            results = self._run_inference(input_tensor)

            # Visualize results
            self._visualize_detection(left_frame, file_path, results)
            
            # Update object list
            self._update_object_list(right_frame, results)

        except Exception as e:
            print(f"Detection error: {e}")

    def _load_image_to_numpy(self, path):
        """Optimized image to numpy conversion."""
        image = Image.open(path).convert('RGB')
        return np.array(image)

    def _run_inference(self, input_tensor):
        """Perform model inference with error handling."""
        try:
            # Ensure model is loaded
            if self.model is None:
                raise ValueError("Model not loaded")

            # Run inference
            return self.model(input_tensor)
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def _visualize_detection(self, frame, image_path, results):
        """Create visualization of detected objects."""
        if results is None:
            return

        # Load image
        input_image = Image.open(image_path)
        
        # Create figure
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(input_image)

        # Extract detection results
        bboxes = results['detection_boxes'][0].numpy()
        labels = results['detection_classes'][0].numpy()
        scores = results['detection_scores'][0].numpy()

        # Process and draw detections
        height, width = np.array(input_image).shape[:2]
        for i in range(len(scores)):
            if scores[i] > 0.5:
                bbox = bboxes[i]
                y1, x1, y2, x2 = bbox[0]*height, bbox[1]*width, bbox[2]*height, bbox[3]*width
                
                # Draw rectangle
                rect = plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1, 
                    fill=False, edgecolor='red', linewidth=2
                )
                ax.add_patch(rect)

                # Add label
                label = self.category_index.get(int(labels[i]), {'name': 'Unknown'})
                ax.text(
                    x1, y1, 
                    f"{label['name']} {scores[i]:.2f}", 
                    color='white', 
                    backgroundcolor='red'
                )

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _update_object_list(self, frame, results):
        """Update list of detected objects."""
        if results is None:
            return

        # Clear existing widgets
        for widget in frame.winfo_children():
            widget.destroy()

        # Title
        title = tk.Label(frame, text="Detected Objects", font=("Arial", 16))
        title.pack(pady=10)

        # Extract and display objects
        labels = results['detection_classes'][0].numpy()
        scores = results['detection_scores'][0].numpy()

        for i in range(len(scores)):
            if scores[i] > 0.5:
                label = self.category_index.get(int(labels[i]), {'name': 'Unknown'})
                obj_label = tk.Label(
                    frame, 
                    text=f"{label['name']} ({scores[i]:.2f})", 
                    font=("Arial", 12)
                )
                obj_label.pack(anchor='w')

    def _open_file(self):
        """Placeholder for file open functionality."""
        print("Open file functionality to be implemented")

    def _show_about(self):
        """Placeholder for about dialog."""
        print("TensorFlow Object Detection Inference Panel v1.0")

def main():
    app = ObjectDetectionApp()
    app.mainloop()

if __name__ == "__main__":
    main()
