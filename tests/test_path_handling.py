import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import config

def test_path_handling():
    assert config.PROJECT_ROOT == Path(__file__).parent.parent
    assert config.MODELS_DIR == Path(__file__).parent.parent / "models"
    assert config.DATASETS_DIR == Path(__file__).parent.parent / "data_sets"
    assert config.SCRIPTS_DIR == Path(__file__).parent.parent / "scripts"
    assert config.TESTS_DIR == Path(__file__).parent.parent / "tests"
    assert config.TRAINING_DIR == Path(__file__).parent.parent / "training"
    assert config.IMAGE_DATA_DIR == Path(__file__).parent.parent / "data_sets" / "image_data"
    assert config.MASK_PERSON_TEST_DIR == Path(__file__).parent.parent / "data_sets" / "image_data" / "mask_person_test"
    assert config.RANDOM_OBJECTS_DIR == Path(__file__).parent.parent / "data_sets" / "image_data" / "random_objects"   
    assert config.VIDEO_DATA_DIR == Path(__file__).parent.parent / "data_sets" / "video_data"
    assert config.YOLOV8_MASK == Path(__file__).parent.parent / "models" / "yolov8n_mask.pt"
    assert config.YOLOV8_PERSON == Path(__file__).parent.parent / "models" / "yolov8n_person.pt"
    assert config.YOLOV11_MASK == Path(__file__).parent.parent / "models" / "yolov11n_mask.pt"
    assert config.YOLOV11_PERSON == Path(__file__).parent.parent / "models" / "yolov11n_person.pt"

test_path_handling()

