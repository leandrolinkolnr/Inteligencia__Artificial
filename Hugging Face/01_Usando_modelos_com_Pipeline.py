from transformers import pipeline


# A função pipeline da biblioteca transformers do Hugging Face é uma ferramenta de alto nível que
#  simplifica o uso de modelos  para várias tarefas de processamento de linguagem natural (NLP). 


# A função pipeline pode ser usada para várias tarefas de NLP, como:

# Geração de Texto (text-generation): Completar ou gerar texto com base em um prompt.
# Preenchimento de Máscaras (fill-mask): Preencher palavras ou frases ausentes em um texto.
# Classificação de Sentimentos (sentiment-analysis): Determinar o sentimento (positivo, negativo, etc.) de um texto.
# Resposta a Perguntas (question-answering): Responder a perguntas baseadas em um contexto fornecido.
# Tradução de Idiomas (translation): Traduzir texto de um idioma para outro.
# Reconhecimento de Entidades Nomeadas (ner): Identificar e classificar entidades como pessoas, organizações, datas, etc., em um texto.
# Entre outras tarefas.


modelo = pipeline('fill-mask')  # Chamando o modelo que quero usar
predicoes = modelo.predict('The capital of <mask> is Lisboa.')   # Fazendo a predição de acordo com o modelo
print(predicoes)

for predicao in predicoes:          # Personalizando a visualização
    resposta = predicao['token_str']
    score = predicao['score']
    frase = predicao['sequence']
    print(f'Predição "{resposta}" com score {(score * 100):.2f}% -> "{frase}"')

