import os

import dotenv
import requests

dotenv.load_dotenv()

modelo = 'google/gemma-7b-it'
url = f"https://api-inference.huggingface.co/models/{modelo}"

json = {
    'inputs': 'Olá, qual o seu nome?',
    'options': {'use_cache': False, 'wait_for_model': True},
}

token = os.environ['HF_TOKEN']
headers = {'Authorization': f'Bearer {token}'}

response = requests.post(url, json=json, headers=headers)
print(response.json())
