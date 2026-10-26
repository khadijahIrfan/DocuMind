
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

print("Testing Gemini API...")
start = time.perf_counter()

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with exactly: Gemini is working.",
        config={"http_options": {"timeout": 120000}},
    )

    print(f"Elapsed time: {time.perf_counter() - start:.2f} seconds")
    print("Response:", response.text)

except Exception as error:
    print(f"Elapsed time: {time.perf_counter() - start:.2f} seconds")
    print("Error type:", type(error).__name__)
    print("Error:", error)
