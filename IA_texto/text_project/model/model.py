import pandas as pd
import ast
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer

def preprocess_text(text):
    if isinstance(text, str):  # Verificar si text es una cadena de texto
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres no alfabéticos
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenizar el texto
        tokens = word_tokenize(text)
        # Eliminar stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Stemming (opcional, puedes omitirlo si no lo deseas)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        # Unir tokens en una cadena
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''  # Si no es una cadena de texto, retornar una cadena vacía

# Crear el pipeline
pipeline = Pipeline([
    # Paso de preprocesamiento de texto
    ('preprocess', FunctionTransformer(preprocess_text)),
    # Convertir texto a vectores usando CountVectorizer (o cualquier otro vectorizador que desees usar)
    ('vectorize', CountVectorizer())
])

def cargar_datos(nombre_archivo):
    # Cargar el CSV
    df = pd.read_csv(nombre_archivo)

    # Convertir la columna 'info' de cadena a diccionario
    df['info'] = df['info'].apply(ast.literal_eval)

    # Expandir el diccionario en columnas separadas
    df = pd.concat([df.drop(['info'], axis=1), df['info'].apply(pd.Series)], axis=1)

    # Seleccionar las columnas relevantes ('post', 'summaries', etc.)
    df = df[['post', 'summaries']]

    df['post'] = df['post'].astype(str)

    return df

# Ruta relativa al archivo CSV
nombre_archivo = '../data/comparisons_train.csv'

df = cargar_datos(nombre_archivo)

# Aplicar el pipeline al texto
X_text = pipeline.fit_transform(df['post'])

# Ahora X_text contiene el texto preprocesado y convertido a vectores
print(X_text)