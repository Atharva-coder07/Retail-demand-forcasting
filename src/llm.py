import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Add it to a local .env file (never hardcode it).")

genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"  # fast + cheap, right fit for narration tasks

def get_model():
    """Returns a configured Gemini GenerativeModel instance."""
    return genai.GenerativeModel(MODEL_NAME)
