from openai import OpenAI
from pydantic import BaseModel
import pandas as pd

from determine_relevant_features import determine_relevant_features
from csv_pruning import prune

client = OpenAI(api_key="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")

OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

s = "XXXXXXX"
# Define the features dictionary
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
    41: "torso tilt",  # Ensure this only appears once
    42: "neck tilt",
    43: "left ankle dorsiflexion",
    44: "right ankle dorsiflexion",
    45: "ankle symmetry"
}

# Update the output model to include frame-level analysis as well as overall summary
class FormOutput(BaseModel):
    frame_analysis: str
    overall_issue: str
    overall_advice: str

def interpret_csv(exercise_name, model_landmarks_csv, user_landmarks_csv,
                  model_features_csv, user_features_csv,
                  model_rom_csv, user_rom_csv,
                  feature_dict):
    # Determine which features are relevant for the exercise
    relevant_features = determine_relevant_features(exercise_name)
    
    # Prune each CSV to include only the relevant features
    model_landmarks_pruned = prune(model_landmarks_csv, relevant_features)
    user_landmarks_pruned = prune(user_landmarks_csv, relevant_features)
    model_features_pruned = prune(model_features_csv, relevant_features)
    user_features_pruned = prune(user_features_csv, relevant_features)
    
    # Convert the pruned CSVs to string representations
    model_landmarks_data = model_landmarks_pruned.to_csv(index=False)
    user_landmarks_data = user_landmarks_pruned.to_csv(index=False)
    model_features_data = model_features_pruned.to_csv(index=False)
    user_features_data = user_features_pruned.to_csv(index=False)
    
    # Construct a system prompt that instructs a frame-by-frame analysis
    system_prompt = (
        "You are a highly experienced exercise biomechanics analyst. Your task is to analyze detailed movement data from CSV files "
        "for exercises on a frame-by-frame basis. The CSV data includes landmark positions, computed joint angles, and symmetry measures for each frame. "
        "For every frame, compare the user's performance with the model's performance. "
        "Identify any discrepancies in joint angles and symmetry—such as if one arm is moving correctly while the other is not. "
        "Also incorporate range-of-motion (ROM) information if provided. "
        "Then, summarize the overall issues and provide actionable advice. "
        "Avoid duplicate references to any body part (for example, do not mention 'torso' twice)."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Exercise Name: {exercise_name}"},
        {"role": "user", "content": f"Extra Feature Descriptions:\n{feature_dict}"},
        {"role": "user", "content": f"Model Landmark CSV Data:\n{model_landmarks_data}"},
        {"role": "user", "content": f"User Landmark CSV Data:\n{user_landmarks_data}"},
        {"role": "user", "content": f"Model Frame Features CSV Data:\n{model_features_data}"},
        {"role": "user", "content": f"User Frame Features CSV Data:\n{user_features_data}"},
        {"role": "user", "content": f"Model ROM CSV Data:\n{model_rom_csv}"},
        {"role": "user", "content": f"User ROM CSV Data:\n{user_rom_csv}"}
    ]
    
    # Send the message list to the OpenAI chat model
    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=FormOutput
    )
    
    return {
        "frame_analysis": response.choices[0].message.parsed.frame_analysis,
        "overall_issue": response.choices[0].message.parsed.overall_issue,
        "overall_advice": response.choices[0].message.parsed.overall_advice
    }

# Note: The exercise name is now set to "Lat pull down" to match your video.
response = interpret_csv(
    "Lat pull down",                    # Corrected exercise name
    "landmarks.csv",                      # Model landmark CSV
    "landmarks_user.csv",                 # User landmark CSV
    "frame_features.csv",                 # Model extra features CSV
    "frame_features_user.csv",            # User extra features CSV
    "rom_summary.csv",                    # Model ROM summary CSV
    "rom_summary_user.csv",               # User ROM summary CSV
    features
)

print(response)
