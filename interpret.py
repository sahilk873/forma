from openai import OpenAI
from pydantic import BaseModel

from determine_relevant_features import determine_relevant_features
from csv_pruning import prune

client = OpenAI(api_key="")
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

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


features = {
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
    32: "right foot index",
    33: "left elbow angle",
    34: "right elbow angle",
    35: "left knee angle",
    36: "right knee angle",
    37: "elbow symmetry",
    38: "knee symmetry",
    39: "shoulder symmetry",
    40: "hip symmetry",
    41: "torso tilt",
    42: "neck tilt",
    43: "left ankle dorsiflexion",
    44: "right ankle dorsiflexion",
    45: "ankle symmetry"
}

class FormOutput(BaseModel):
    body_part: str
    issue: str
    advice: str

def interpret_csv(exercise_name, model_landmarks_csv, user_landmarks_csv,
                  model_features_csv, user_features_csv,
                  model_rom_csv, user_rom_csv,
                  feature_dict):
    
    relevant_features = determine_relevant_features(exercise_name)

    model_landmarks_csv = prune(model_landmarks_csv, relevant_features).to_csv()
    user_landmarks_csv = prune(user_landmarks_csv, relevant_features).to_csv()
    model_features_csv = prune(model_features_csv, relevant_features).to_csv()
    user_features_csv = prune(user_features_csv, relevant_features).to_csv()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly experienced exercise biomechanics analyst. "
                "Your task is to analyze detailed movement data from CSV files for exercises. "
                "The CSV data contains landmark positions, computed joint angles, and symmetry measures multiple frames. \n\n"
                "Each numeric value in the CSV files has a specific meaning:\n"
                "- Joint angles (e.g., left_elbow_angle, right_knee_angle) indicate the degree of bend at that joint.\n"
                "- Symmetry measures (e.g., elbow_symmetry, shoulder_symmetry) represent the absolute differences between the left and right sides; "
                "smaller numbers indicate more balanced movement.\n"
                "A wider range may imply overextension or inconsistent movement patterns.\n\n"
                "Tell the user how their movement differs from the model and how they can improve. "),
        },
        {"role": "user", "content": f"Exercise Name: {exercise_name}"},
        {"role": "user", "content": f"Extra Feature Descriptions:\n{feature_dict}"},
        {"role": "user", "content": f"Model Landmark CSV Data:\n{model_landmarks_csv}"},
        {"role": "user", "content": f"User Landmark CSV Data:\n{user_landmarks_csv}"},
        {"role": "user", "content": f"Model Frame Features CSV Data:\n{model_features_csv}"},
        {"role": "user", "content": f"User Frame Features CSV Data:\n{user_features_csv}"},
        {"role": "user", "content": f"Body Parts Dictionary:\n{body_parts}"}
        
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


response = interpret_csv(
    "Cable pull down",
    "landmarks.csv",  # Model landmark CSV
    "landmarks_user.csv",             # User landmark CSV
    "frame_features.csv",             # Model extra features CSV
    "frame_features_user.csv",             # User extra features CSV
    "rom_summary.csv",                # Model ROM summary CSV
    "rom_summary_user.csv",                # User ROM summary CSV
    features
)

print(response)
