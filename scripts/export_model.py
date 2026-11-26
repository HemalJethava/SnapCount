# Exports YOLO11n to float32 TFLite and copies it to assets/yolo11n_float32.tflite
# Use Python 3.10-3.12 (TensorFlow does not support newer versions):
#   pip install ultralytics tensorflow
#   python scripts/export_model.py
import shutil
from pathlib import Path

from ultralytics import YOLO

path = YOLO("yolo11n.pt").export(format="tflite", imgsz=640)  # -> yolo11n_saved_model/yolo11n_float32.tflite
src = Path(path) if str(path).endswith(".tflite") else Path("yolo11n_saved_model/yolo11n_float32.tflite")
dst = Path(__file__).resolve().parent.parent / "assets" / "yolo11n_float32.tflite"
dst.parent.mkdir(exist_ok=True)
shutil.copy(src, dst)
print("Model copied to", dst)
