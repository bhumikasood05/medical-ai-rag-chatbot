from groq import Groq
import os
from dotenv import load_dotenv

# -----------------------
# LOAD ENV VARIABLES
# -----------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# -----------------------
# CREATE GROQ CLIENT
# -----------------------
client = Groq(api_key=api_key)

# -----------------------
# LLM FUNCTION
# -----------------------
def generate_answer(context, question):

    prompt = f"""
You are a STRICT medical information assistant.

RULES:
- Use ONLY the provided context
- Do NOT guess or assume anything
- Do NOT give diagnosis or medical advice
- If answer is not in context, say: "Not found in dataset"
- Keep answer short (2-4 lines max)

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content