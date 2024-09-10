from transformers import pipeline


chatbot = pipeline(
    "text-generation",
    model="Felladrin/Llama-68M-Chat-v1",   # O modelo escolhido no Hugging Face
    max_new_tokens=300,
    penalty_alpha=0.5,                      # Parametros recomendados pelo prorpio Modelo
    top_k=4,
)


mensagem_sistema = 'You are a helpful artificial intelligence assistant.'
conversa = mensagem_sistema

while True:
    mensagem_usuario = input('Escreva sua pergunta (em inglês): ')
    conversa += f'<|im_start|>user\n{mensagem_usuario}<|im_end|>\n<|im_start|>assistant'   # Seguindo padrao do modelo
    resposta = chatbot(conversa)
    conversa = resposta[0]['generated_text']
    resposta_formatada = conversa.split('<|im_start|>assistant\n')[-1].rstrip('<|im_end|>')
    print(f'Resposta do bot: {resposta_formatada}')
