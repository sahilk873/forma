from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key="")  # Set your API key

OPENAI_MODEL = "gpt-4o-2024-08-06"

body_parts = {
    0: "nose",
    1: "left eye (inner)",
    2: "left eye",
    3: "left eye (outer)",
    4: "right eye (inner)",
    5: "right eye",
    6: "right eye (outer)",
    7: "left ear",
    8: "right ear",
    9: "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"
}

# Expanded extra features (including new joint angles and symmetry measures)
feature_descriptions = {
    "left_elbow_angle": "Angle at the left elbow between the left shoulder, left elbow, and left wrist",
    "right_elbow_angle": "Angle at the right elbow between the right shoulder, right elbow, and right wrist",
    "left_knee_angle": "Angle at the left knee between the left hip, left knee, and left ankle",
    "right_knee_angle": "Angle at the right knee between the right hip, right knee, and right ankle",
    "elbow_symmetry": "Absolute difference between left and right elbow angles",
    "knee_symmetry": "Absolute difference between left and right knee angles",
    "shoulder_symmetry": "Absolute angle (in degrees) of the line connecting the shoulders relative to horizontal",
    "hip_symmetry": "Absolute angle (in degrees) of the line connecting the hips relative to horizontal",
    "torso_tilt": "Angle (in degrees) between the line connecting the midpoints of the shoulders and hips relative to vertical",
    "neck_tilt": "Angle (in degrees) between vertical and the line connecting the neck (midpoint between shoulders) and the nose",
    "left_ankle_dorsiflexion": "Angle at the left ankle between the left knee, left ankle, and left foot index",
    "right_ankle_dorsiflexion": "Angle at the right ankle between the right knee, right ankle, and right foot index",
    "ankle_symmetry": "Absolute difference between left and right ankle dorsiflexion angles"
}

class FormOutput(BaseModel):
    body_part: str
    issue: str
    advice: str

def interpret_csv(exercise_name, model_landmarks_csv, user_landmarks_csv,
                  model_features_csv, user_features_csv,
                  model_rom_csv, user_rom_csv,
                  feature_dict):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant specializing in analyzing exercise movement data. "
                "Compare a user's movement data against the model movement data for the exercise "
                f"{exercise_name}. In addition to landmark positions, analyze extra features such as joint angles, "
                "symmetry measures, and range of motion (ROM) metrics across multiple frames. "
                "Identify discrepancies in key body parts and provide clear, actionable feedback for form improvement."
            ),
        },
        {"role": "user", "content": f"Exercise Name: {exercise_name}"},
        {"role": "user", "content": f"Model Landmark CSV Data:\n{model_landmarks_csv}"},
        {"role": "user", "content": f"User Landmark CSV Data:\n{user_landmarks_csv}"},
        {"role": "user", "content": f"Model Frame Features CSV Data:\n{model_features_csv}"},
        {"role": "user", "content": f"User Frame Features CSV Data:\n{user_features_csv}"},
        {"role": "user", "content": f"Model ROM Summary CSV Data:\n{model_rom_csv}"},
        {"role": "user", "content": f"User ROM Summary CSV Data:\n{user_rom_csv}"},
        {"role": "user", "content": f"Body Parts Dictionary:\n{body_parts}"},
        {"role": "user", "content": f"Extra Feature Descriptions:\n{feature_dict}"}
    ]

    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=FormOutput
    )
    return {
        "issue": response.choices[0].message.parsed.issue,
        "advice": response.choices[0].message.parsed.advice
    }

# Example usage:
response = interpret_csv(
    "Cable pull down",
    "landmarks_with_features.csv",  # Model landmark CSV
    "landmarks_left.csv",             # User landmark CSV
    "frame_features.csv",             # Model extra features CSV
    "frame_features.csv",             # User extra features CSV
    "rom_summary.csv",                # Model ROM summary CSV
    "rom_summary.csv",                # User ROM summary CSV
    feature_descriptions
)
print(response)
