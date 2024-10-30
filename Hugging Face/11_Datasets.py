# Manipulação básica de um Dataset

from datasets import load_dataset

dataset = load_dataset("imdb")   # Tem treino, teste e não supervisionado
print(dataset)

dataset_treino = dataset['train']   # Pegando so o treino
print(dataset_treino)

print(dataset_treino[9])   # Linha completa
print(dataset_treino[9]['label'])   # Pegando dado especifico 
print(dataset_treino['label'])    # Label = 0 e 1  (Positivo e negativo)

#Convertendo para um DataFrame do pandas
df = dataset_treino.to_pandas()
