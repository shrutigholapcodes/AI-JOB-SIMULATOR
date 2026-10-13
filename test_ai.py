from google import genai

print("STEP 1")

client = genai.Client()

print("STEP 2")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say only: Hello"
)

print("STEP 3")

print(response.text)

print("STEP 4")