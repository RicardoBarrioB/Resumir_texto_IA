import operator
from collections import defaultdict
from time import time

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralCoclustering


def number_normalizer(tokens): #Cuando exista algún token numérico, la función revisa si el primer caracter es un número y si es así lo sustituye por #NUMBER(marcador de posición)
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(TfidfVectorizer): #Crea un tokenizador, convierte un doc de texto a una matriz TF-IDF
    def build_tokenizer(self): #modifica el metodo super para personalizar el tokenizador
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

vectorizer = NumberNormalizingVectorizer()

corpus = [
    "El gato come pescado.",
    "Los perros corren en el parque.",
    "Hoy es un buen día."
]

# Transformar los datos de texto en una matriz TF-IDF
X = vectorizer.fit_transform(corpus)

# Imprimir el vocabulario generado por el vectorizador
print("Vocabulary:", vectorizer.vocabulary_)

# Imprimir la matriz TF-IDF resultante
print("TF-IDF Matrix:")
print(X.toarray())