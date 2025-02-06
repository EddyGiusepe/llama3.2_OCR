#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro
"""
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Loading variables from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
