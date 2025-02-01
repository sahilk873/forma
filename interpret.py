from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key="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")
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


class FormOutput(BaseModel):
    body_part: str
    issue: str
    advice: str


def interpret_csv(exercise_name, model_csv, user_csv):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant specializing in analyzing exercise movement data. "
                "Compare a user's movement data (user_csv) against the model movement data (model_csv) based on the provided body_parts dictionary while performing the exercise {exercise_name}. "
                "Identify discrepancies in key body parts and provide clear, actionable feedback to correct the user's form. "
                "Your feedback should be specific, describing both the observed issue and advice on how to fix it."
                "Your feedback should comment the entire movement, not just the start and end points, including commenting on the relationships between body parts."
            ),
        },
        {"role": "user", "content": f"Exercise Name: {exercise_name}"},
        {"role": "user", "content": f"Model CSV Data:\n{model_csv}"},
        {"role": "user", "content": f"User CSV Data:\n{user_csv}"},
        {"role": "user", "content": f"Body Parts Dictionary:\n{body_parts}"}
    ]

    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=FormOutput
    )
    return {"issue": response.choices[0].message.parsed.issue, "advice": response.choices[0].message.parsed.advice}

response = interpret_csv("Cable pull down", "landmarks.csv", "landmarks_left.csv")
print(response)
 