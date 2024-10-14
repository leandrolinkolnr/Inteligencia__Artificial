
import requests


# Usando um modelo de resumo em português


modelo = "csebuetnlp/mT5_multilingual_XLSum"
url = f"https://api-inference.huggingface.co/models/{modelo}"

with open('noticia.txt', encoding = 'utf-8') as f:   
    texto = f.read()     # Abrindo o arquivo e passando conteudo para a variavel texto

json = {
    'inputs': texto,
    'parameters': {'min_length': 100},  # Em tokens (Não resuma exageradamente)
    'options': {'use_cache': False, 'wait_for_model': True},
}

response = requests.post(url, json=json) # Capturando resposta

print(response.json())
