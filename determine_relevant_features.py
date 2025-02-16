from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict

import os
from dotenv import load_dotenv

load_dotenv()
 
api_key = os.getenv("OPENAI_KEY")


client = OpenAI(api_key=api_key)

OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

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


class RelevantFeatures(BaseModel):
    features_list: List[int]


def determine_relevant_features(exercise_name: str) -> List[int]:
    messages = []
    messages.append({
        "role": "system",
        "content": ("You are an AI that selects relevant body part features for a given exercise. "
                    "You will receive a dictionary where the keys are numbers representing body parts and motion features, "
                    "and the values are the corresponding names. Your task is to return a list of the numerical keys that are relevant for the provided exercise. Return atleast two motion features each time which dictionary numbers 33-45 related to the exercise alongside every body parts that is relevant.")
    })
    messages.append({
        "role": "user",
        "content": f"Here is the dictionary of features:\n{features}\n\nGiven the exercise '{exercise_name}', "
                   "return a list of the feature keys (numbers) that are relevant. For exercises that use the arms, make sure to return keys relating to the shoulder, wrist, and elbow."
    })

    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=RelevantFeatures
    )

    return response.choices[0].message.parsed.features_list
