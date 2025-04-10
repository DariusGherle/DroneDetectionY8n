from ultralytics import YOLO
import cv2
import yt_dlp

def get_youtube_stream(url):
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def main():
    # Load YOLO model
    model = YOLO("runs/detect/drone_detector_v12/weights/best.pt")

    #YouTube video URL
    yt_url = "https://www.youtube.com/watch?v=myyC8T7Jbsw&t=1368s"
    stream_url = get_youtube_stream(yt_url)

    # Open video stream
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Failed to open YouTube video stream.")
        return
    print("Streaming YouTube video\n Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or dropped.")
            break

        # Run detection
        results = model.predict(frame, conf=0.4, verbose=False)
        annotated = results[0].plot()

        # Show annotated frame
        cv2.imshow("Drone Detector - YouTube Stream", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
