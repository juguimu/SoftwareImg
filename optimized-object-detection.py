import tkinter as tk
from typing import Dict, Any
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import logging

class ObjectDetectionApp:
    def __init__(self, model_path: str, image_path: str):
        """
        Initialize the Object Detection Application
        
        Args:
            model_path (str): Path to the pre-trained TensorFlow model
            image_path (str): Path to the input image
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load COCO category index
        self.category_index = self._load_coco_categories()

        # Initialize TensorFlow model with error handling
        try:
            self.model = self._load_tensorflow_model(model_path)
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise

        # Load and process input image
        self.image_path = image_path
        self.image_np = self._load_image_to_numpy(image_path)

        # Initialize main application window
        self.root = tk.Tk()
        self.root.title("TensorFlow Object Detection")
        self.root.geometry('1024x768')

        # Perform object detection
        self.detection_results = self._detect_objects()

        # Create UI
        self._create_ui()

    @staticmethod
    def _load_coco_categories() -> Dict[int, Dict[str, Any]]:
        """
        Load COCO dataset category index
        
        Returns:
            Dict of category mappings
        """
        # Truncated to most common categories for performance
        return {
            1: {'id': 1, 'name': 'person'},
            2: {'id': 2, 'name': 'bicycle'},
            3: {'id': 3, 'name': 'car'},
            # ... (add more categories as needed)
        }

    def _load_tensorflow_model(self, model_path: str):
        """
        Load TensorFlow SavedModel with error handling
        
        Args:
            model_path (str): Path to the saved model
        
        Returns:
            Loaded TensorFlow model
        """
        try:
            # Clear previous sessions for consistent performance
            tf.keras.backend.clear_session()
            
            # Use optimized model loading
            model = tf.saved_model.load(model_path)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_image_to_numpy(self, path: str) -> np.ndarray:
        """
        Efficiently convert image to numpy array
        
        Args:
            path (str): Image file path
        
        Returns:
            Numpy array of the image
        """
        try:
            with Image.open(path) as img:
                # Convert to RGB to ensure 3 channels
                img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Image loading failed: {e}")
            raise

    def _detect_objects(self):
        """
        Perform object detection on the loaded image
        
        Returns:
            Detection results
        """
        try:
            # Optimize input tensor preparation
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(self.image_np, 0), 
                dtype=tf.float32
            )
            
            # Perform detection with performance optimization
            detections = self.model(input_tensor)
            
            return {
                'boxes': detections['detection_boxes'][0].numpy(),
                'classes': detections['detection_classes'][0].numpy().astype(np.int32),
                'scores': detections['detection_scores'][0].numpy()
            }
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return None

    def _create_ui(self):
        """
        Create the application user interface
        """
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib figure for image with detections
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image_np)
        
        # Draw detections
        self._draw_detections(ax)

        # Embed matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _draw_detections(self, ax, confidence_threshold: float = 0.5):
        """
        Draw bounding boxes and labels for detected objects
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes to draw on
            confidence_threshold (float): Minimum confidence to display detection
        """
        height, width, _ = self.image_np.shape
        
        for i, score in enumerate(self.detection_results['scores']):
            if score < confidence_threshold:
                continue
            
            # Get bounding box coordinates
            box = self.detection_results['boxes'][i]
            ymin, xmin, ymax, xmax = box
            
            # Convert to pixel coordinates
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            # Get class
            class_id = int(self.detection_results['classes'][i])
            class_name = self.category_index.get(class_id, {}).get('name', 'Unknown')
            
            # Draw rectangle
            rect = plt.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin, 
                fill=False, 
                edgecolor='red', 
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                xmin, 
                ymin, 
                f'{class_name}: {score:.2f}', 
                color='white', 
                backgroundcolor='red'
            )

    def run(self):
        """
        Run the application main loop
        """
        self.root.mainloop()

# Example usage
if __name__ == '__main__':
    app = ObjectDetectionApp(
        model_path='path/to/centernet_model', 
        image_path='path/to/input/image.jpg'
    )
    app.run()
