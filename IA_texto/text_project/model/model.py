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
        # Eliminar caracteres no alfabéticos excepto el apóstrofe
        text = re.sub(r'[^a-zA-Z\s\']', '', text)
        # Tokenizar el texto
        tokens = word_tokenize(text)
        # Eliminar stop words
        #stop_words = set(stopwords.words('english'))
        #tokens = [word for word in tokens if word not in stop_words]
        return tokens  # Devolver lista de tokens preprocesados
    else:
        return []  # Si no es una cadena de texto, retornar una cadena vacía

# Crear el pipeline
pipeline = Pipeline([
    ('preprocess', FunctionTransformer(preprocess_text)),
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

    # Procesar el contenido del texto antes de pasarlo al pipeline
    df['post'] = df['post'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else '')

    return df

# Ruta relativa al archivo CSV
nombre_archivo = '../datas/comparisons_train.csv'

# Cargar los datos
df = cargar_datos(nombre_archivo)

# Ahora aplicar el pipeline al texto
X_text = pipeline.fit_transform(df['post'])

# Ahora X_text contiene el texto preprocesado y convertido a vectores
print(X_text)