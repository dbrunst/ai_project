from openai import OpenAI
import json

def generate():
  client = OpenAI()
  json_content = ''

  with open("raw_data.json", "r") as file:
    json_content = json.load(file)

  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": """
              Given a hash of ids and text, return a json object with the id and an array of only people's names without prefixes but including capitalized suffixes.
              Capitalize name appropriately
              """
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": json.dumps(json_content)
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": """
              To extract only the people's names from the given text, we should look for names typically formatted with a first and last name, separated by a space or comma.
              The input hash contains some examples of such names that can be extracted:
                \n\n```json\n{\n  \"1\": [\"Zeke Miller III\", \"Aamer Madhani\", \"Daniella Genovese\"],\n  \"2\": [],\n  \"3\": [\"john smith\"]\n}\n```\n\n
              This output correctly identifies and returns arrays of unique names from each string in the input hash.
              It removes non-name text such as dates, publisher information, professional titles, and prefixes such as dr.
              """
          }
        ]
      }
    ],
    response_format={
      "type": "json_object"
    },
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return json.loads(response.choices[0].message.content)

print(json.dumps(generate(), indent=4))