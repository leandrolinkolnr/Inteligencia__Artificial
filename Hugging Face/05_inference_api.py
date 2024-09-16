import requests   # Necessario realizar requisições
from transformers import AutoTokenizer


# A Inference API permite a utilização de modelos hospedados no HuggingFace diretamente via uma interface de API.


chat = [
    {"role": "user", "content": "Olá, qual o seu nome?"},
    {"role": "assistant", "content": "Olá, eu sou um modelo de AI. Como posso ajudar?"},
    {"role": "user", "content": "Gostaria de aprender Python. Você tem alguma dica?"},
]

# Tokenizer é uma ferramenta que converte texto em tokens, unidades menores para ser processadas por modelos .
# AutoTokenizer.from_pretrained: carrega um tokenizer pré-treinado rápidamente, garantindo compatibilidade com o modelo.

# Carregam o tokenizer para o modelo e aplicam um template de chat, preparando-o para a geração de respostas.

tokenizer_mixtral = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')   # Instanciando com modelo escolhido
template_mixtral = tokenizer_mixtral.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print('----- Chat formatado para modelo Mixtral -----')
print(template_mixtral)




# ================ Resposta ajustada com uso de templating ================ 

modelo = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

chat = [
    {"role": "user", "content": "Hello, what is your name?"},
]

tokenizer = AutoTokenizer.from_pretrained(modelo)
chat_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

url = f"https://api-inference.huggingface.co/models/{modelo}"

json = {
    'inputs': chat_str,
    'options': {'use_cache': False, 'wait_for_model': True},
}

response = requests.post(url, json=json)
print(response.json())



# ================ Resposta ajustada com uso de templating em loop ================ 


modelo = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(modelo)   # Instanciando o tokenizador
url = f"https://api-inference.huggingface.co/models/{modelo}"
chat = []
while True:
    mensagem = input('Faça sua pergunta em inglês ("q" para sair):')
    if mensagem == 'q':
        break
    chat.append({'role': 'user', 'content': mensagem})  # Adicionando mensagem a conversa
    # Aplica um template de chat ao texto fornecido e estrutura a entrada com um formato que facilita a interação do modelo.
    chat_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) 
    json = {
        'inputs': chat_str,
        'parameters': {'max_new_tokens': 1_000},  # Posso passar parametros que o modelo aceita
        'options': {'use_cache': False, 'wait_for_model': True},
    }
    response = requests.post(url, json=json).json()
    mensagem_chatbot = response[0]['generated_text'].split('[/INST]')[-1]
    print('Resposta do chatbot:', mensagem_chatbot)
    chat.append({'role': 'assistant', 'content': mensagem_chatbot})

print(chat)
