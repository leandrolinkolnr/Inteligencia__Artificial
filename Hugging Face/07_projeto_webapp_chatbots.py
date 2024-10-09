import os

import dotenv
import requests
import streamlit as st
from transformers import AutoTokenizer

dotenv.load_dotenv()
token = os.environ['HF_TOKEN']


# "INST" e "<start_of_turn>model" são tokens especiais usados para marcar ou delimitar partes específicas do texto 
# durante a tokenização. Esses tokens são geralmente usados para indicar instruções ou iniciar uma nova parte do diálogo

modelos = {
    'mistralai/Mixtral-8x7B-Instruct-v0.1': '[/INST]',
    'google/gemma-7b-it': '<start_of_turn>model\n',
}

nome_modelo = st.selectbox('Selecione um modelo:', options=modelos)   # Selecionando um dos dois modelos
token_modelo = modelos[nome_modelo]

url = f"https://api-inference.huggingface.co/models/{nome_modelo}"   # Pegando a url com o modelo escolhido

tokenizer = AutoTokenizer.from_pretrained(nome_modelo)  # Carrega um tokenizador pré-treinado, preparando o texto para ser 
                                                        # processado por um modelo de linguagem, convertendo texto em tokens

# Definindo o modelo
if ('modelo_atual' not in st.session_state              # Primeira execução do programa
    or st.session_state['modelo_atual'] != nome_modelo  # Outro modelo escolhido
):
    # Reiniciar o chat e altera o  modelo atual
    st.session_state['mensagens'] = []
    st.session_state['modelo_atual'] = nome_modelo


mensagens = st.session_state['mensagens']

area_chat = st.empty()
pergunta_usuario = st.chat_input('Faça sua pergunta aqui: ')

if pergunta_usuario:     # SE tiver pergunta
    mensagens.append({'role': 'user', 'content': pergunta_usuario})   # Adicione nas mensagens
    # Aplicando um templat de chat e preparando as mensagens para gerar uma resposta,
    inputs = tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    json = {
        'inputs': inputs,           # Rreparando os dados e as configurações para uma requisição a uma API. 
        'parameters': {'max_new_tokens': 1_000},  
        'options': {'use_cache': False, 'wait_for_model': True},
    }
    headers = {
        'Authentication': f'Bearer {token}',   # Fornece o token necessário para autenticar a requisição.
    }




response = requests.post(url, json=json, headers=headers).json()   # Envia uma requisição POST para uma API com dados
resposta_chatbot = response[0]['generated_text'].split(token_modelo)[-1]   # Captura resposta
mensagens.append({'role': 'assistant', 'content': resposta_chatbot})  # Adciona a resposta nas mensagens


# Função para aparecer as mensagens no chat de forma simples
with area_chat.container():
    for mensagem in mensagens:
        chat = st.chat_message(mensagem['role'])
        chat.markdown(mensagem['content'])
