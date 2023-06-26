import dotenv
from dotenv import load_dotenv
import os


def load_environment_keys(debug: bool = True):
    e = dotenv.find_dotenv()
    if debug:
        print(f"environment found at {e}")
    has_env: bool = load_dotenv(e, verbose=True, override=True)
    if not has_env:
        print("Did not found environment file, using system OpenAI key (if exists)")
    openai_key = os.getenv('OPENAI_API_KEY')
    return openai_key