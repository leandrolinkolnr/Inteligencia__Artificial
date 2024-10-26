from transformers import pipeline
import requests

# Analise de Sentimentos (Simples)


modelo = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
classificador = pipeline("text-classification", model=modelo)

reviews = [
    "Até então não tenho do que reclamar. Estou usando pra estudo, está bem tranquilo até aqui.",
    "Acho que vale o custo benefício, caso seja para usos básicos.",
    "A bateria do notebook está descarregando muito rápido.",
    "Eu estava com muito medo de me arrepender da compra. Mas eu realmente gostei! Ótimo demais, comprem!",
    "Muito bom, recomendo!",
    "Não comprem, caro demais pelo que oferece.",
    "Super custo benefício, pelo preço que paguei superou todas as minhas expectativas.",
    "Excelente, zero arrependimentos. Muito muito bom.",
    "Esperava um pouco mais. Mas é um produto bom. Não coloquei mais estrelas pois não usei direito.",
]

for review in reviews:
    print(f'Avaliação: "{review}"')
    resultado = classificador(review)
    prob = resultado[0]['score'] * 100
    print('Análise de sentimento:', resultado[0]['label'], f'{prob:.2f}%')






# Top_k = Mostra a probabilidade de todas (Positivo, neutro e negativo


review = "Acho que vale o custo benefício, caso seja para usos básicos."
resultado = classificador(review, top_k=None)

print(review)
print('Análise de sentimento:', resultado)






# Modelo de classificação de emoções (em inglês)


classificador = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions")

frases = [
    'I am feeling a bit better, thanks for asking!',
    'Well, it could be worse.',
    'This is awful!',
    'I am very happy with the test results!',
    'Unbelievable!',
    'Go away!!!',
]


for frase in frases:
    print(f'frase: "{frase}"')
    resultado = classificador(frase)
    prob = resultado[0]['score'] * 100
    print('Análise de sentimento:', resultado[0]['label'], f'{prob:.2f}%')







# Modelo de análise de sentimento em frases de contexto financeiro.


url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
frases = [
    'However, the growth margin slowed down due to the financial crisis.',
    'The company laid off thousands of employees last week.',
    'According to their updated strategy for the years 2009-2012, the company targets a long-term net sales growth',
    'Result before taxes decreased to nearly EUR 14.5mn, compared to nearly EUR 20mn in the previous accounting period.'
]

for frase in frases:
    json = {
        'inputs': frase,
        'options': {'use_cache': False, 'wait_for_model': True},
    }
    response = requests.post(url, json=json)
    print('Frase original:', frase)
    print(response.json(), '\n')







# Modelo de análise de ironia em tweets


detector_ironia = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony", top_k=None)

for frase in [
    "@Mountgrace lol i know! its so frustrating isnt it?!",
    "I don't like clowns but I'm going to be one.",
    "Now I remember why I buy books online @user #servicewithasmile",
    "Simply having a wonderful christmas time :D",
]:
    print(f'frase: "{frase}"')
    resultado = detector_ironia(frase)
    resultado_mais_provavel = resultado[0][0]
    prob = resultado_mais_provavel['score'] * 100
    print('Análise de sentimento:', resultado_mais_provavel['label'], f'{prob:.2f}%')


