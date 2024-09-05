from transformers import pipeline


modelos = [
    {
        'nome': 'FacebookAI/xlm-roberta-base',
        'token': '<mask>',
    },
    {
        'nome': 'neuralmind/bert-base-portuguese-cased',
        'token': '[MASK]',
    },
    {
        'nome': 'rufimelo/Legal-BERTimbau-base',
        'token': '[MASK]',
    },
]

for dict_modelo in modelos:
    nome_modelo = dict_modelo['nome']
    print(f'Testando modelo {nome_modelo}...')
    token = dict_modelo['token']
    modelo = pipeline('fill-mask', model=nome_modelo)
    frase = f'Este documento é essencial para a {token}.'
    predicoes = modelo.predict(frase)
    for predicao in predicoes:
        resposta = predicao['token_str']
        score = predicao['score']
        frase = predicao['sequence']
        print(f'Predição "{resposta}" com score {(score * 100):.2f}% -> "{frase}"')
    input('Aperte "Enter" para seguir para o próximo modelo')
