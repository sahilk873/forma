import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dictionary mapping feature names to numeric keys
feature_name_to_key = {
    "left_elbow_angle": 33,
    "right_elbow_angle": 34,
    "left_knee_angle": 35,
    "right_knee_angle": 36,
    "elbow_symmetry": 37,
    "knee_symmetry": 38,
    "shoulder_symmetry": 39,
    "hip_symmetry": 40,
    "torso_tilt": 41,
    "neck_tilt": 42,
    "left_ankle_dorsiflexion": 43,
    "right_ankle_dorsiflexion": 44,
    "ankle_symmetry": 45
}

# Descriptions of extra features (for reference)
feature_descriptions = {
    "left_elbow_angle": "Angle at the left elbow between the left shoulder, left elbow, and left wrist",
    "right_elbow_angle": "Angle at the right elbow between the right shoulder, right elbow, and right wrist",
    "left_knee_angle": "Angle at the left knee between the left hip, left knee, and left ankle",
    "right_knee_angle": "Angle at the right knee between the right hip, right knee, and right ankle",
    "elbow_symmetry": "Absolute difference between left and right elbow angles",
    "knee_symmetry": "Absolute difference between left and right knee angles",
    "shoulder_symmetry": (
        "Absolute angle (in degrees) of the line connecting the left and right shoulders relative to the horizontal axis; "
        "a value near 0 indicates well-aligned shoulders"
    ),
    "hip_symmetry": (
        "Absolute angle (in degrees) of the line connecting the left and right hips relative to the horizontal axis; "
        "a value near 0 indicates well-aligned hips"
    ),
    "torso_tilt": (
        "Angle (in degrees) between the line connecting the midpoints of the shoulders and hips relative to the vertical axis; "
        "lower values indicate a more upright posture"
    ),
    "neck_tilt": (
        "Angle (in degrees) between the vertical axis and the line connecting the approximate neck (midpoint between shoulders) and the nose; "
        "indicates head tilt"
    ),
    "left_ankle_dorsiflexion": (
        "Angle at the left ankle between the left knee, left ankle, and left foot index, indicating ankle dorsiflexion"
    ),
    "right_ankle_dorsiflexion": (
        "Angle at the right ankle between the right knee, right ankle, and right foot index, indicating ankle dorsiflexion"
    ),
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

# Helper function: Calculate angle (in degrees) at point b given points a, b, and c
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_video(
    video_path: str,
    output_video_path: str,
    rom_csv_path: str,
    landmarks_csv_path: str,
    features_csv_path: str
):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Prepare a dictionary to store keypoints (x, y, z, visibility)
    landmark_data = {}
    # Prepare a list for frame-level features
    frame_features_data = []
    
    # Define keypoints and connections
    relevant_keypoints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
    relevant_connections = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 25), (24, 26),
        (25, 27), (26, 28)
    ]
    
    # Video properties for writing output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

            # Normalization for scale invariance: compute midpoints and torso length
            left_shoulder_orig = (landmarks[11].x * w, landmarks[11].y * h)
            right_shoulder_orig = (landmarks[12].x * w, landmarks[12].y * h)
            left_hip_orig = (landmarks[23].x * w, landmarks[23].y * h)
            right_hip_orig = (landmarks[24].x * w, landmarks[24].y * h)
            mid_shoulder = ((left_shoulder_orig[0] + right_shoulder_orig[0]) / 2,
                            (left_shoulder_orig[1] + right_shoulder_orig[1]) / 2)
            mid_hip = ((left_hip_orig[0] + right_hip_orig[0]) / 2,
                       (left_hip_orig[1] + right_hip_orig[1]) / 2)
            torso_length = np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip)) + 1e-6

            def normalize_point(point):
                return ((point[0] - mid_hip[0]) / torso_length, (point[1] - mid_hip[1]) / torso_length)

            # Compute normalized coordinates for key landmarks
            norm_left_shoulder = normalize_point(left_shoulder_orig)
            norm_right_shoulder = normalize_point(right_shoulder_orig)
            norm_left_elbow = normalize_point((landmarks[13].x * w, landmarks[13].y * h))
            norm_right_elbow = normalize_point((landmarks[14].x * w, landmarks[14].y * h))
            norm_left_wrist = normalize_point((landmarks[15].x * w, landmarks[15].y * h))
            norm_right_wrist = normalize_point((landmarks[16].x * w, landmarks[16].y * h))
            norm_left_hip = normalize_point(left_hip_orig)
            norm_right_hip = normalize_point(right_hip_orig)
            norm_left_knee = normalize_point((landmarks[25].x * w, landmarks[25].y * h))
            norm_right_knee = normalize_point((landmarks[26].x * w, landmarks[26].y * h))
            norm_left_ankle = normalize_point((landmarks[27].x * w, landmarks[27].y * h))
            norm_right_ankle = normalize_point((landmarks[28].x * w, landmarks[28].y * h))
            norm_left_foot_index = normalize_point((landmarks[31].x * w, landmarks[31].y * h))
            norm_right_foot_index = normalize_point((landmarks[32].x * w, landmarks[32].y * h))
            norm_nose = normalize_point((landmarks[0].x * w, landmarks[0].y * h))

            # Draw connections for visualization
            for connection in relevant_connections:
                start_idx, end_idx = connection
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                start_coords = (int(start_lm.x * w), int(start_lm.y * h))
                end_coords = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_coords, end_coords, (0, 255, 255), 2)

            # Save landmark data for selected keypoints
            for idx_key in relevant_keypoints:
                lm = landmarks[idx_key]
                if (idx_key, "x") not in landmark_data:
                    landmark_data[(idx_key, "x")] = []
                    landmark_data[(idx_key, "y")] = []
                    landmark_data[(idx_key, "z")] = []
                    landmark_data[(idx_key, "visibility")] = []
                landmark_data[(idx_key, "x")].append(lm.x)
                landmark_data[(idx_key, "y")].append(lm.y)
                landmark_data[(idx_key, "z")].append(lm.z)
                landmark_data[(idx_key, "visibility")].append(lm.visibility)
                # Optionally draw circles for visualization
                if idx_key in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Compute features using normalized coordinates
            left_elbow_angle = calculate_angle(norm_left_shoulder, norm_left_elbow, norm_left_wrist)
            right_elbow_angle = calculate_angle(norm_right_shoulder, norm_right_elbow, norm_right_wrist)
            left_knee_angle = calculate_angle(norm_left_hip, norm_left_knee, norm_left_ankle)
            right_knee_angle = calculate_angle(norm_right_hip, norm_right_knee, norm_right_ankle)
            elbow_symmetry = abs(left_elbow_angle - right_elbow_angle)
            knee_symmetry = abs(left_knee_angle - right_knee_angle)
            dx_shoulder = norm_right_shoulder[0] - norm_left_shoulder[0]
            dy_shoulder = norm_right_shoulder[1] - norm_left_shoulder[1]
            shoulder_angle = np.degrees(np.arctan2(dy_shoulder, dx_shoulder))
            shoulder_symmetry = abs(shoulder_angle)
            dx_hip = norm_right_hip[0] - norm_left_hip[0]
            dy_hip = norm_right_hip[1] - norm_left_hip[1]
            hip_angle = np.degrees(np.arctan2(dy_hip, dx_hip))
            hip_symmetry = abs(hip_angle)
            norm_mid_shoulder = ((norm_left_shoulder[0] + norm_right_shoulder[0]) / 2,
                                 (norm_left_shoulder[1] + norm_right_shoulder[1]) / 2)
            torso_tilt = abs(np.degrees(np.arctan2(norm_mid_shoulder[0], norm_mid_shoulder[1])))
            norm_neck = norm_mid_shoulder
            dx_neck = norm_nose[0] - norm_neck[0]
            dy_neck = norm_nose[1] - norm_neck[1]
            neck_line_angle = np.degrees(np.arctan2(dx_neck, dy_neck))
            neck_tilt = abs(neck_line_angle)
            left_ankle_dorsiflexion = calculate_angle(norm_left_knee, norm_left_ankle, norm_left_foot_index)
            right_ankle_dorsiflexion = calculate_angle(norm_right_knee, norm_right_ankle, norm_right_foot_index)
            ankle_symmetry = abs(left_ankle_dorsiflexion - right_ankle_dorsiflexion)

            # Store computed features for this frame
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

            # Update ROM tracker
            update_rom("left_elbow_angle", left_elbow_angle)
            update_rom("right_elbow_angle", right_elbow_angle)
            update_rom("left_knee_angle", left_knee_angle)
            update_rom("right_knee_angle", right_knee_angle)
            update_rom("left_ankle_dorsiflexion", left_ankle_dorsiflexion)
            update_rom("right_ankle_dorsiflexion", right_ankle_dorsiflexion)
            update_rom("neck_tilt", neck_tilt)
            update_rom("torso_tilt", torso_tilt)

            # Optionally, add text overlays (this can be removed if not needed)
            cv2.putText(frame, f"LElbow: {int(left_elbow_angle)}",
                        (int(landmarks[13].x * w), int(landmarks[13].y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"RElbow: {int(right_elbow_angle)}",
                        (int(landmarks[14].x * w), int(landmarks[14].y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"LKnee: {int(left_knee_angle)}",
                        (int(landmarks[25].x * w), int(landmarks[25].y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"RKnee: {int(right_knee_angle)}",
                        (int(landmarks[26].x * w), int(landmarks[26].y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"NeckTilt: {int(neck_tilt)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"TorsoTilt: {int(torso_tilt)}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"LAnkleDF: {int(left_ankle_dorsiflexion)}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
            cv2.putText(frame, f"RAnkleDF: {int(right_ankle_dorsiflexion)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

        # Write frame to output and display (optional)
        out.write(frame)
        #cv2.imshow("Pose Estimation with Features", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save the frame's features
        frame_features_data.append(frame_features)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -----------------------------------------------------------------------------------------
    # Save Landmark Data
    # -----------------------------------------------------------------------------------------
    df_landmarks = pd.DataFrame.from_dict(
        landmark_data,
        orient="index",
        columns=[f"frame_{i}" for i in range(frame_idx)]
    )
    df_landmarks.reset_index(inplace=True)
    df_landmarks.rename(columns={"level_0": "landmark_id", "level_1": "coordinate"}, inplace=True)
    df_landmarks.to_csv(landmarks_csv_path, index=False)

    # -----------------------------------------------------------------------------------------
    # Save Frame Features (transposed)
    # -----------------------------------------------------------------------------------------
    df_features = pd.DataFrame(frame_features_data)
    rename_mapping = {}
    for col in df_features.columns:
        if col == "frame":
            continue
        if col in feature_name_to_key:
            rename_mapping[col] = feature_name_to_key[col]
        else:
            rename_mapping[col] = col
    df_features.rename(columns=rename_mapping, inplace=True)
    df_features.set_index("frame", inplace=True)
    df_features = df_features.transpose().reset_index()
    df_features.rename(columns={"index": "feature"}, inplace=True)
    old_frame_cols = df_features.columns[1:]
    new_frame_cols = [f"frame_{int(col)}" for col in old_frame_cols]
    df_features.columns = ["feature"] + new_frame_cols
    df_features.to_csv(features_csv_path, index=False)

    # -----------------------------------------------------------------------------------------
    # Save ROM Summary
    # -----------------------------------------------------------------------------------------
    rom_summary = []
    for feature, values in rom_tracker.items():
        rom_summary.append({
            "feature": feature,
            "min": values["min"],
            "max": values["max"]
        })
    df_rom = pd.DataFrame(rom_summary)
    df_rom.to_csv(rom_csv_path, index=False)

    print("Landmark data saved to", landmarks_csv_path)
    print("Frame-level features saved to", features_csv_path)
    print("Range of Motion summary saved to", rom_csv_path)
    print("Extra Feature Descriptions:")
    for key, desc in feature_descriptions.items():
        print(f"{key}: {desc}")

# Example usage:
# process_video(
#     video_path="videos/IMG_4622.mov",
#     output_video_path="videos/processed_video_with_features.mp4",
#     rom_csv_path="rom_summary_user.csv",
#     landmarks_csv_path="landmarks_user.csv",
#     features_csv_path="frame_features_user.csv"
# )
