from groq import Groq

from config import GROQ_API_KEY

_CLIENT=Groq(api_key=GROQ_API_KEY)

def get_llm():
    return _CLIENT