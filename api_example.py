from dotenv import load_dotenv  # pip install python-dotenv
import os

load_dotenv()

#OPENALEX_API_KEY="someAPIstring"
api_key = os.getenv("OPENALEX_API_KEY")