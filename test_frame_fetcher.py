import cv2
from scripts.frame_fetcher import FrameFetcher  # Import your FrameFetcher class

def main():
    fetcher = FrameFetcher("data_sets/video_data/3205619-hd_1920_1080_25fps.mp4")  # Use webcam (change to file path for video)
    
    while True:
        ret, frame = fetcher.get_frame()
        if not ret:
            continue  # Skip if no frame is available

        cv2.imshow("Frame", frame)  # Display frame
        cv2.waitKey(500)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    fetcher.stop()  # Stop frame fetching
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()