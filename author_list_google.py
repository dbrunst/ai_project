import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting



generation_config = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]

def generate():
    vertexai.init(project="gemini-playground-441622", location="us-central1")
    textsi_1 = """
      Given a hash of ids and text, return a json object with the id and an array of only people's names without prefixes but including capitalized suffixes.
      Capitalize name appropriately
      """

    model = GenerativeModel(
        "gemini-1.0-pro-002",
        system_instruction=[textsi_1]
    )


    with open("raw_data.json", "r") as file:
      text1 = json.dumps(json.load(file))

    responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    raw = ''
    for response in responses:
        raw += response.text

    raw = raw.replace("```json\n", "").replace("\n```", "")

    raw = json.loads(raw)
    print(json.dumps(raw, indent=4))

generate()