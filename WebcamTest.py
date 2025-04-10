from ultralytics import YOLO
import cv2

def main():
    model = YOLO("runs/detect/drone_detector_v12/weights/best.pt")

    cap = cv2.VideoCapture(0)  # Use 1 or 2 if 0 doesn't work

    if not cap.isOpened():
        print("Could not open webcam. Try a different index (e.g. 1 or 2).")
        return

    print("Webcam opened successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model.predict(source=frame, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Drone Detector - Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
