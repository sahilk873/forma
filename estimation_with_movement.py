import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the video
video_path = "sahilnormal.mov"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare a list to store landmark data
landmark_data = []

# Define the keypoints and connections to track (gym-related body parts)
relevant_keypoints = [
    11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
]
relevant_connections = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 25), (24, 26), (25, 27), (26, 28)
]

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "processed_video.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Threshold for movement detection
movement_threshold = 5000  # Adjust this value if needed

# Initialize variables for movement detection
prev_gray = None
movement_detected = False
movement_started = False
frame_idx = 0
movement_start_frame = 0
movement_end_frame = 0

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Convert to grayscale for movement detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        # Calculate frame difference
        frame_diff = cv2.absdiff(prev_gray, gray_frame)
        movement_score = np.sum(frame_diff)

        # Detect the start and end of movement
        if movement_score > movement_threshold:
            if not movement_started:
                movement_start_frame = frame_idx
                movement_started = True
            movement_detected = True
            movement_end_frame = frame_idx
        else:
            movement_detected = False

    prev_gray = gray_frame

    # Process only during detected movement
    if movement_detected or movement_started:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Draw pose landmarks and connections for relevant keypoints
            for connection in relevant_connections:
                start_idx, end_idx = connection
                start_landmark = results.pose_landmarks.landmark[start_idx]
                end_landmark = results.pose_landmarks.landmark[end_idx]

                # Convert normalized coordinates to pixel values
                h, w, _ = frame.shape
                start_coords = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_coords = (int(end_landmark.x * w), int(end_landmark.y * h))

                # Draw connection line
                cv2.line(frame, start_coords, end_coords, (0, 255, 255), 2)

            for idx in relevant_keypoints:
                landmark = results.pose_landmarks.landmark[idx]
                x = landmark.x
                y = landmark.y
                z = landmark.z
                visibility = landmark.visibility

                # Append the relevant landmark data with frame and landmark index
                landmark_data.append({
                    "frame": frame_idx,
                    "landmark_id": idx,
                    "x": x,
                    "y": y,
                    "z": z,
                    "visibility": visibility
                })

                # Draw circle for the keypoint
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Draw circle on the landmark

        # Write the processed frame to the output video
        out.write(frame)

        # Display the video frame with landmarks
        cv2.imshow("Pose Estimation", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Convert landmark data to a DataFrame and save it as a CSV file
df = pd.DataFrame(landmark_data)
df.to_csv("landmarks.csv", index=False)

print(f"Landmark data saved to landmarks.csv")
print(f"Processed video saved to {output_video_path}")
print(f"Movement detected from frame {movement_start_frame} to {movement_end_frame}")
