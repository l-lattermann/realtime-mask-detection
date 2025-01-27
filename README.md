# YOLO Model Comparison for Mask and Person Detection

## **Project Purpose**
This project compares the performance of different YOLO models in detecting:
1. **Persons** and **face masks**.
2. **Social distancing** by measuring distances between detected persons.

The goal is to evaluate model efficiency and accuracy under real-world conditions.

---

## **Installation**
Follow these steps to set up the project:

1. Clone this repository:
   ```bash
   git clone https://github.com/l-lattermann/yolo-mobilenet-realtime-mask-detection.git
   cd <repository_directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Ensure YOLO models are available in the `models` directory. You can download pre-trained models from [Ultralytics](https://github.com/ultralytics/yolov5).

---

## **Project Structure**
- **`data_sets/`**  
  Contains images and video datasets for testing models.

- **`models/`**  
  Stores YOLO model weights and configurations.

- **`scripts/`**  
  Includes utility scripts for inference and video processing.

- **`train_runs/`**  
  Results and logs from model training and testing.

- **`.gitignore`**  
  Specifies files and directories to be ignored by Git.

- **`detect_yolo.ipynb`**  
  Notebook for performing detections using the YOLO models.

- **`scratches_SSD.ipynb`**  
  Notebook for initial training experiments.

- **`train_yolo_colab_nb.ipynb`**  
  Notebook for training YOLO models on Google Colab.

- **`understand_yolo_tensor_structure.ipynb`**  
  Notebook for exploring YOLO tensor outputs.
