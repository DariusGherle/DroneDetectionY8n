from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # Model pre-antrenat

    model.train(
        data="C:/Users/40754/PycharmProjects/DroneDetWithAudio/VisionDataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="drone_detector_v1",
        device=0  #GPU
    )

if __name__ == "__main__":
    main()
