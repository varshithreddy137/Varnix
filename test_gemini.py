
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

start = time.perf_counter()

interaction = client.interactions.create(
    model=model,
    input="Reply with only the word OK.",
    generation_config={
        "thinking_level": "low"
    },
)

elapsed = time.perf_counter() - start

print("Response:", interaction.output_text)
print(f"Latency: {elapsed:.2f} seconds")
