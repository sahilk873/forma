from openai import OpenAI
from pydantic import BaseModel

from determine_relevant_features import determine_relevant_features
from csv_pruning import prune

client = OpenAI(api_key="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")  # Set your API key

OPENAI_MODEL = "o1-mini-2024-09-12"

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

feature_descriptions = {
    "left_elbow_angle": "Angle at the left elbow between the left shoulder, left elbow, and left wrist",
    "right_elbow_angle": "Angle at the right elbow between the right shoulder, right elbow, and right wrist",
    "left_knee_angle": "Angle at the left knee between the left hip, left knee, and left ankle",
    "right_knee_angle": "Angle at the right knee between the right hip, right knee, and right ankle",
    "elbow_symmetry": "Absolute difference between left and right elbow angles; lower values indicate more symmetrical movement",
    "knee_symmetry": "Absolute difference between left and right knee angles; lower values indicate better symmetry",
    "shoulder_symmetry": (
        "The absolute angle (in degrees) of the line connecting the shoulders relative to the horizontal axis. "
        "Values closer to 0 imply well-aligned shoulders, while higher values indicate misalignment."
    ),
    "hip_symmetry": (
        "The absolute angle (in degrees) of the line connecting the hips relative to the horizontal axis. "
        "Values near 0 indicate good alignment, while larger values show misalignment."
    ),
    "torso_tilt": (
        "Angle (in degrees) between the line connecting the midpoints of the shoulders and hips and the vertical axis. "
        "Lower values suggest a more upright posture; higher values indicate excessive tilt."
    ),
    "neck_tilt": (
        "Angle (in degrees) between vertical and the line connecting the neck (midpoint between shoulders) and the nose. "
        "A higher number indicates that the head is tilted excessively forward or backward."
    ),
    "left_ankle_dorsiflexion": (
        "Angle at the left ankle between the left knee, left ankle, and left foot index. "
        "A higher angle suggests greater dorsiflexion, which can indicate overextension or improper form."
    ),
    "right_ankle_dorsiflexion": (
        "Angle at the right ankle between the right knee, right ankle, and right foot index. "
        "Similar to the left side, larger values may indicate excessive dorsiflexion."
    ),
    "ankle_symmetry": "Absolute difference between left and right ankle dorsiflexion angles; lower values are better."
}

class FormOutput(BaseModel):
    body_part: str
    issue: str
    advice: str

def interpret_csv(exercise_name,
                  model_landmarks_csv,
                  user_landmarks_csv,
                  model_features_csv,
                  user_features_csv,
                  model_rom_csv,
                  user_rom_csv,
                  feature_dict):
    relevant_features = determine_relevant_features(exercise_name)

    model_landmarks_csv = prune(model_landmarks_csv, relevant_features).to_csv()
    user_landmarks_csv = prune(user_landmarks_csv, relevant_features).to_csv()
    model_features_csv = prune(model_features_csv, relevant_features).to_csv()
    user_features_csv = prune(user_features_csv, relevant_features).to_csv()

    prompt = f"""
System Instructions:
You are a highly experienced exercise biomechanics analyst. Your task is to analyze 
detailed movement data from CSV files for a Cable Pull Down exercise. The CSV data 
contains landmark positions, computed joint angles, symmetry measures, and range-of-
motion (ROM) metrics over multiple frames.

Each numeric value in the CSV files has a specific meaning:
- Joint angles (e.g., left_elbow_angle, right_knee_angle) indicate the degree of bend 
  at that joint.
- Symmetry measures (e.g., elbow_symmetry, shoulder_symmetry) represent the absolute 
  differences between the left and right sides; smaller numbers indicate more balanced 
  movement.
- ROM metrics provide the minimum and maximum values for specific joint angles across 
  the exercise repetition. A wider range may imply overextension or inconsistent 
  movement patterns.

For this exercise, the data shows that the left side of the body and arm is descending 
too far, while the right side is not descending enough. Additionally, shoulder symmetry 
is off. Please analyze these discrepancies in detail, citing specific CSV data metrics 
where applicable.

Provide a comprehensive, multi-layered analysis that includes:
1. Detailed explanations of movement discrepancies, with special attention to left/
   right imbalances.
2. An in-depth discussion of joint angle differences (elbow, knee, shoulder, neck,
   ankle) and their implications.
3. A thorough analysis of symmetry issues, particularly the misalignment of the
   shoulders.
4. A summary of range-of-motion variations across frames.
5. Specific, actionable advice for form correction, referencing the relevant metrics.
Your response should be long, comprehensive, and directly address the specific issues 
observed in the data.

User Instructions:
Exercise Name: {exercise_name}

Model Landmark CSV Data:
{model_landmarks_csv}

User Landmark CSV Data:
{user_landmarks_csv}

Model Frame Features CSV Data:
{model_features_csv}

User Frame Features CSV Data:
{user_features_csv}

Model ROM Summary CSV Data:
{model_rom_csv}

User ROM Summary CSV Data:
{user_rom_csv}

Body Parts Dictionary:
{body_parts}

Extra Feature Descriptions:
{feature_dict}

Please analyze the CSV data thoroughly and provide a comprehensive report. In your 
report, include the following:
1. A detailed explanation of discrepancies in movement between the user's data and 
   the model's data, specifically highlighting any imbalances (for example, the left 
   side descending too far while the right side does not move sufficiently).
2. An in-depth discussion of joint angle differences, including elbow, knee, shoulder, 
   neck, and ankle angles, and what these numerical values imply about the user's form.
3. A detailed analysis of symmetry issues (e.g., shoulder and ankle symmetry) and what 
   the numeric values indicate about alignment.
4. A summary of the range-of-motion (ROM) metrics, explaining what the min and max 
   values reveal about the consistency and quality of the movement.
5. Specific, actionable advice on how to correct these issues, with reference to the 
   numerical metrics from the CSV data.

Ensure that your response is extensive, data-driven, and provides a clear roadmap for 
form improvement.
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    # Here you can implement any parsing logic you prefer, for instance if the model
    # returns JSON, you can parse it with BaseModel. Otherwise, you may just return
    # the raw text or do a custom parse. Below is a simple raw text return:
    return response.choices[0].message.content

response = interpret_csv(
    "Cable pull down",               # exercise_name
    "landmarks.csv",                 # model_landmarks_csv
    "landmarks_user.csv",            # user_landmarks_csv
    "frame_features.csv",            # model_features_csv
    "frame_features_user.csv",       # user_features_csv
    "rom_summary.csv",               # model_rom_csv
    "rom_summary_user.csv",          # user_rom_csv
    feature_descriptions             # feature_dict
)

print(response)
