from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    model.train(data="data/data.yaml", epochs=25, imgsz=160, batch=64, lr0=0.0005)
    res = model.val(data="data/data.yaml", save_json=True, plots=True)
    #plot_results("runs/detect/exp/results.csv", dir="runs/detect/exp")