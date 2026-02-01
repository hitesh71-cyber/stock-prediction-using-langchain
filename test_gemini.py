from google import genai
import os

key = os.getenv("GOOGLE_API_KEY")
print("KEY FOUND:", bool(key))

client = genai.Client(api_key=key)

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Reply with the word OK only"
)

print(response.text)
