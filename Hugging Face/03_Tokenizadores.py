from transformers import AutoTokenizer, AutoModel

nome_modelo = 'FacebookAI/xlm-roberta-base'

modelo = AutoModel.from_pretrained(nome_modelo)
tokenizador = AutoTokenizer.from_pretrained(nome_modelo)

print(modelo)
print(tokenizador)

tokens = tokenizador('A linguagem <mask> é uma ferramenta inovadora.')
print(tokens)

inputs = tokenizador('A linguagem <mask> é uma ferramenta inovadora.', return_tensors='pt')
print(inputs)

outputs = modelo(**inputs)
print(outputs)
