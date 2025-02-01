import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the video
video_path = "IMG_4622.mov"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare a list to store landmark data
landmark_data = []

# Define the keypoints to track (indices for gym-related body parts)
relevant_keypoints = [
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip
    24,  # Right hip
    25,  # Left knee
    26,  # Right knee
    27,  # Left ankle
    28,  # Right ankle
]

# Filter connections to draw only relevant body lines
relevant_connections = [
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist
    (11, 23),  # Left shoulder to left hip
    (12, 24),  # Right shoulder to right hip
    (23, 25),  # Left hip to left knee
    (24, 26),  # Right hip to right knee
    (25, 27),  # Left knee to left ankle
    (26, 28),  # Right knee to right ankle
]

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "processed_video_left.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

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
df.to_csv("landmarks_left.csv", index=False)

print("Landmark data saved")
print(f"Processed video saved to {output_video_path}")
