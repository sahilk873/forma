import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the video (update with your video path)
video_path = "IMG_4622.mov"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare lists to store per-landmark data and per-frame feature data
landmark_data = []
frame_features_data = []

# Include additional keypoints needed for extra features (nose, foot indices)
relevant_keypoints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

# Connections to draw (using the original ones for visualization)
relevant_connections = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28)
]

# Helper function: Calculate angle (in degrees) at point b given points a, b, and c
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Dictionary to describe extra features
feature_descriptions = {
    "left_elbow_angle": "Angle at the left elbow between the left shoulder, left elbow, and left wrist",
    "right_elbow_angle": "Angle at the right elbow between the right shoulder, right elbow, and right wrist",
    "left_knee_angle": "Angle at the left knee between the left hip, left knee, and left ankle",
    "right_knee_angle": "Angle at the right knee between the right hip, right knee, and right ankle",
    "elbow_symmetry": "Absolute difference between left and right elbow angles",
    "knee_symmetry": "Absolute difference between left and right knee angles",
    "shoulder_symmetry": ("Absolute angle (in degrees) of the line connecting the left and right shoulders relative to the horizontal axis; "
                          "a value near 0 indicates well-aligned shoulders"),
    "hip_symmetry": ("Absolute angle (in degrees) of the line connecting the left and right hips relative to the horizontal axis; "
                     "a value near 0 indicates well-aligned hips"),
    "torso_tilt": ("Angle (in degrees) between the line connecting the midpoints of the shoulders and hips relative to the vertical axis; "
                   "lower values indicate a more upright posture"),
    "neck_tilt": ("Angle (in degrees) between the vertical axis and the line connecting the approximate neck (midpoint between shoulders) and the nose; "
                  "indicates head tilt"),
    "left_ankle_dorsiflexion": ("Angle at the left ankle between the left knee, left ankle, and left foot index, indicating ankle dorsiflexion"),
    "right_ankle_dorsiflexion": ("Angle at the right ankle between the right knee, right ankle, and right foot index, indicating ankle dorsiflexion"),
    "ankle_symmetry": "Absolute difference between left and right ankle dorsiflexion angles"
}

# Define which angle features we want to track the range of motion (ROM) for
rom_features = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
    "neck_tilt", "torso_tilt"
]
rom_tracker = {feature: {"min": None, "max": None} for feature in rom_features}

def update_rom(feature_name, value):
    if value is None:
        return
    if rom_tracker[feature_name]["min"] is None or value < rom_tracker[feature_name]["min"]:
        rom_tracker[feature_name]["min"] = value
    if rom_tracker[feature_name]["max"] is None or value > rom_tracker[feature_name]["max"]:
        rom_tracker[feature_name]["max"] = value

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "processed_video_with_features_user.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    frame_features = {"frame": frame_idx}

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Draw connections for visualization
        for connection in relevant_connections:
            start_idx, end_idx = connection
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            start_coords = (int(start_lm.x * w), int(start_lm.y * h))
            end_coords = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(frame, start_coords, end_coords, (0, 255, 255), 2)

        # Save landmark data (for keypoints we care about)
        for idx in relevant_keypoints:
            lm = landmarks[idx]
            landmark_data.append({
                "frame": frame_idx,
                "landmark_id": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
            # Draw circles for the original keypoints (skip extra ones if desired)
            if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # --- Compute standard joint angles ---
        # Elbow angles
        left_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
        right_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
        left_elbow = (landmarks[13].x * w, landmarks[13].y * h)
        right_elbow = (landmarks[14].x * w, landmarks[14].y * h)
        left_wrist = (landmarks[15].x * w, landmarks[15].y * h)
        right_wrist = (landmarks[16].x * w, landmarks[16].y * h)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Knee angles
        left_hip = (landmarks[23].x * w, landmarks[23].y * h)
        right_hip = (landmarks[24].x * w, landmarks[24].y * h)
        left_knee = (landmarks[25].x * w, landmarks[25].y * h)
        right_knee = (landmarks[26].x * w, landmarks[26].y * h)
        left_ankle = (landmarks[27].x * w, landmarks[27].y * h)
        right_ankle = (landmarks[28].x * w, landmarks[28].y * h)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # --- Compute symmetry measures ---
        elbow_symmetry = abs(left_elbow_angle - right_elbow_angle)
        knee_symmetry = abs(left_knee_angle - right_knee_angle)
        # Shoulder symmetry: angle of the line joining the shoulders relative to horizontal
        dx_shoulder = right_shoulder[0] - left_shoulder[0]
        dy_shoulder = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = np.degrees(np.arctan2(dy_shoulder, dx_shoulder))
        shoulder_symmetry = abs(shoulder_angle)  # ideally near 0
        # Hip symmetry: angle of the line joining the hips relative to horizontal
        dx_hip = right_hip[0] - left_hip[0]
        dy_hip = right_hip[1] - left_hip[1]
        hip_angle = np.degrees(np.arctan2(dy_hip, dx_hip))
        hip_symmetry = abs(hip_angle)

        # Torso tilt: angle between the line joining mid-shoulders and mid-hips relative to vertical
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        dx_mid = mid_hip[0] - mid_shoulder[0]
        dy_mid = mid_hip[1] - mid_shoulder[1]
        torso_tilt = abs(np.degrees(np.arctan2(dx_mid, dy_mid)))  # 0 indicates perfectly vertical

        # --- Additional features ---
        # Neck tilt: angle between vertical and the line connecting the approximate neck (midpoint of shoulders) and the nose (landmark 0)
        nose = (landmarks[0].x * w, landmarks[0].y * h)
        neck = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        dx_neck = nose[0] - neck[0]
        dy_neck = nose[1] - neck[1]
        neck_line_angle = np.degrees(np.arctan2(dx_neck, dy_neck))
        neck_tilt = abs(neck_line_angle)  # deviation from vertical

        # Ankle dorsiflexion: using left knee, left ankle, and left foot index (landmark 31)
        left_foot_index = (landmarks[31].x * w, landmarks[31].y * h)
        left_ankle_dorsiflexion = calculate_angle(left_knee, left_ankle, left_foot_index)
        # Right ankle dorsiflexion: using right knee, right ankle, and right foot index (landmark 32)
        right_foot_index = (landmarks[32].x * w, landmarks[32].y * h)
        right_ankle_dorsiflexion = calculate_angle(right_knee, right_ankle, right_foot_index)
        ankle_symmetry = abs(left_ankle_dorsiflexion - right_ankle_dorsiflexion)

        # Update frame-level features dictionary
        frame_features.update({
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "elbow_symmetry": elbow_symmetry,
            "knee_symmetry": knee_symmetry,
            "shoulder_symmetry": shoulder_symmetry,
            "hip_symmetry": hip_symmetry,
            "torso_tilt": torso_tilt,
            "neck_tilt": neck_tilt,
            "left_ankle_dorsiflexion": left_ankle_dorsiflexion,
            "right_ankle_dorsiflexion": right_ankle_dorsiflexion,
            "ankle_symmetry": ankle_symmetry
        })

        # Update ROM tracker for selected angle features
        update_rom("left_elbow_angle", left_elbow_angle)
        update_rom("right_elbow_angle", right_elbow_angle)
        update_rom("left_knee_angle", left_knee_angle)
        update_rom("right_knee_angle", right_knee_angle)
        update_rom("left_ankle_dorsiflexion", left_ankle_dorsiflexion)
        update_rom("right_ankle_dorsiflexion", right_ankle_dorsiflexion)
        update_rom("neck_tilt", neck_tilt)
        update_rom("torso_tilt", torso_tilt)

        # Optionally, draw text on frame for visualization
        cv2.putText(frame, f"LElbow: {int(left_elbow_angle)}", (int(left_elbow[0]), int(left_elbow[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"RElbow: {int(right_elbow_angle)}", (int(right_elbow[0]), int(right_elbow[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"LKnee: {int(left_knee_angle)}", (int(left_knee[0]), int(left_knee[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"RKnee: {int(right_knee_angle)}", (int(right_knee[0]), int(right_knee[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"NeckTilt: {int(neck_tilt)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"TorsoTilt: {int(torso_tilt)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"LAnkleDF: {int(left_ankle_dorsiflexion)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        cv2.putText(frame, f"RAnkleDF: {int(right_ankle_dorsiflexion)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

    # Write processed frame and display
    out.write(frame)
    cv2.imshow("Pose Estimation with Features", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_features_data.append(frame_features)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Save landmark and frame feature data to CSV files
df_landmarks = pd.DataFrame(landmark_data)
df_landmarks.to_csv("landmarks_user.csv", index=False)

df_features = pd.DataFrame(frame_features_data)
df_features.to_csv("frame_features_user.csv", index=False)


# Create a ROM summary DataFrame from the rom_tracker dictionary
rom_summary = []
for feature, values in rom_tracker.items():
    rom_summary.append({
        "feature": feature,
        "min": values["min"],
        "max": values["max"]
    })
df_rom = pd.DataFrame(rom_summary)
df_rom.to_csv("rom_summary_user.csv", index=False)

print("Landmark data saved to landmarks_with_features.csv")
print("Frame-level features saved to frame_features.csv")
print("Range of Motion summary saved to rom_summary.csv")
print("Extra Feature Descriptions:")
for key, desc in feature_descriptions.items():
    print(f"  {key}: {desc}")
