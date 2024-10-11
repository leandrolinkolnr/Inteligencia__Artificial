

# Tradução Simples usando transformers


from transformers import pipeline

modelo = "facebook/mbart-large-50-many-to-many-mmt"
mensagem = "Olá! Estou aprendendo a programar em Python e a usar modelos de inteligência artificial pelo Hugging Face."

tradutor = pipeline("translation", model=modelo)
traducao = tradutor(mensagem, src_lang='pt_XX', tgt_lang='en_XX')

print(traducao)







# Tradução de diferentes frases para diferentes línguas


modelo = "facebook/mbart-large-50-many-to-many-mmt"
mensagens = [
    "Olá! Estou aprendendo a programar em Python e a usar modelos de inteligência artificial pelo Hugging Face.",
    "Vamos nos encontrar às 15h da próxima sexta-feira. Eu acho que todos os meus amigos vão estar lá!",
    "Três tigres tristes comeram três pratos de trigo.",
    "Ser feliz sem motivo é a mais autêntica forma de felicidade.",
]
linguas = [
    'en_XX',
    'es_XX',
    'fr_XX',
]

tradutor = pipeline("translation", model=modelo)

for lingua in linguas:
    print(f'Traduzindo do português para {lingua}...')
    traducoes = tradutor(mensagens, src_lang='pt_XX', tgt_lang=lingua)
    for mensagem, traducao in zip(mensagens, traducoes):
        print(f'Frase original: "{mensagem}"')
        frase_traduzida = traducao['translation_text']
        print(f'Frase em {lingua}: "{frase_traduzida}"')






# Tradução via Inference API  (Usando recursos do Hugging face)


import requests

modelo = "facebook/mbart-large-50-many-to-many-mmt"
url = f"https://api-inference.huggingface.co/models/{modelo}"

mensagens = [
    "Olá! Estou aprendendo a programar em Python e a usar modelos de inteligência artificial pelo Hugging Face.",
    "Vamos nos encontrar às 15h da próxima sexta-feira. Eu acho que todos os meus amigos vão estar lá!",
    "Três tigres tristes comeram três pratos de trigo.",
    "Ser feliz sem motivo é a mais autêntica forma de felicidade.",
]
linguas = [
    'en_XX',
    'es_XX',
    'fr_XX',
]

for lingua in linguas:
    print(f'Traduzindo do português para {lingua}...')
    json = {
        'inputs': mensagens,
        'parameters': {'src_lang': 'pt_XX', 'tgt_lang': lingua},
        'options': {'use_cache': False, 'wait_for_model': True},
    }
    response = requests.post(url, json=json)
    traducoes = response.json()
    for mensagem, traducao in zip(mensagens, traducoes):
        print(f'Frase original: "{mensagem}"')
        frase_traduzida = traducao['translation_text']
        print(f'Frase em {lingua}: "{frase_traduzida}"')
