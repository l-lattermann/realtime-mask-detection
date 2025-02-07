from pathlib import Path

# Define the project root dynamically
PROJECT_ROOT = Path(__file__).parent

# Standard directories
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "data_sets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
TRAINING_DIR = PROJECT_ROOT / "training"

# Dataset Directories
IMAGE_DATA_DIR = DATASETS_DIR / "image_data"
MASK_PERSON_TEST_DIR = IMAGE_DATA_DIR / "mask_person_test"
RANDOM_OBJECTS_DIR = IMAGE_DATA_DIR / "random_objects"
VIDEO_DATA_DIR = DATASETS_DIR / "video_data"

# YOLO Model Paths
YOLOV8_MASK = MODELS_DIR / "yolov8n_mask.pt"
YOLOV8_PERSON = MODELS_DIR / "yolov8n_person.pt"
YOLOV11_MASK = MODELS_DIR / "yolov11n_mask.pt"
YOLOV11_PERSON = MODELS_DIR / "yolov11n_person.pt"

# Ensure directories exist
for directory in [MODELS_DIR, DATASETS_DIR, SCRIPTS_DIR, TESTS_DIR, TRAINING_DIR, IMAGE_DATA_DIR, MASK_PERSON_TEST_DIR, RANDOM_OBJECTS_DIR, VIDEO_DATA_DIR]:
    directory.mkdir(exist_ok=True)