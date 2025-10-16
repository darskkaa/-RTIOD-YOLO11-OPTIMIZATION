from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

model = YOLO("checkpoints/best.pt")

if __name__ == "__main__":
    res = model.val(data="data/data.yaml", save_json=True, plots=True)