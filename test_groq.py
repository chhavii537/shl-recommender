import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello in JSON like: {\"reply\": \"hello\"}"}],
        temperature=0.2,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    print("SUCCESS:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"ERROR: {e}")