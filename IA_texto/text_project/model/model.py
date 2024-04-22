import pandas as pd
import ast
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from keras.layers import TextVectorization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

def cargar_datos(nombre_archivo):
    # Cargar el CSV
    df = pd.read_csv(nombre_archivo)

    # Convertir la columna 'info' de cadena a diccionario
    df['info'] = df['info'].apply(ast.literal_eval)

    # Expandir el diccionario en columnas separadas
    df = pd.concat([df.drop(['info'], axis=1), df['info'].apply(pd.Series)], axis=1)

    # Seleccionar las columnas relevantes ('post', 'summaries', etc.)
    df = df[['post', 'summaries']]

    return df


nombre_archivo = '../datas/comparisons_train.csv'
df = cargar_datos(nombre_archivo)

# Preprocesamiento de texto
def preprocess_text(text):
    if isinstance(text, str):  # Verificar si text es una cadena de texto
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres no alfabéticos excepto el apóstrofe
        text = re.sub(r'[^a-zA-Z\s\']', '', text)
        # Tokenizar el texto
        tokens = word_tokenize(text)
        # Eliminar stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        print("Tokens después de eliminar stopwords:", tokens)
        # Verificar si quedan tokens después de eliminar las stopwords
        if tokens:
            # Devolver una lista con un solo texto preprocesado
            return [' '.join(tokens)]
        else:
            return ['documento_vacío']  # Devolver un texto representativo para un documento vacío
    else:
        return ['']


# Aplicar preprocesamiento de texto a los datos
df['post'] = df['post'].apply(preprocess_text)
df['summaries'] = df['summaries'].apply(preprocess_text)

# Tokenización
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['post'].tolist() + df['summaries'].tolist())

vocab_size = len(tokenizer.word_index) + 1

# Convertir texto a secuencias de tokens
X = tokenizer.texts_to_sequences(df['post'])
y = tokenizer.texts_to_sequences(df['summaries'])

# Pad sequences para que tengan la misma longitud
maxlen = 100 # ajustar según la longitud máxima de tu texto
X_padded = pad_sequences(X, padding='post', maxlen=maxlen)
y_padded = pad_sequences(y, padding='post', maxlen=maxlen)

# Dividir datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# Construir modelo LSTM
embedding_dim = 100 # ajustar según la dimensión de embedding deseada
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(LSTM(100, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
model.fit(X_train, np.expand_dims(y_train, axis=-1), epochs=10, batch_size=32, validation_split=0.2)

# Evaluar modelo
loss, accuracy = model.evaluate(X_test, np.expand_dims(y_test, axis=-1))
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Guardar modelo
model.save('lstm_model.h5')


