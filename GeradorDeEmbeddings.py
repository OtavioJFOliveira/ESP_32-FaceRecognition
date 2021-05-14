from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np


def load_face(filename):
    # Carregando a imagem do arquivo
    image = Image.open(filename)

    # converter para RGB
    image = image.convert("RGB")

    return asarray(image)


# Carregando as faces de um diretorio
def load_faces(directory_src):
    faces = list()

    # Iterando arquivos
    for filename in listdir(directory_src):
        path = directory_src + filename

        try:
            faces.append(load_face(path))
        except:
            print("Erro na imagem {}".format(path))

    return faces


## Carregando todo o dataset de imagens de faces

def load_fotos(directory_src):
    X, y = list(), list()
    # iterar as pastas por classe
    for subdir in listdir(directory_src):

        path = directory_src + subdir + '\\'

        if not isdir(path):
            continue

        faces = load_faces(path)

        labels = [subdir for _ in range(len(faces))]

        # sumarizar progresso
        print('>Carregadas %d faces da classe: %s' % (len(faces), subdir))

        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)

trainX, trainy = load_fotos(directory_src = "C:\\Users\\otavi\\Desktop\\Mestrado\\Imagens\\Faces\\")

print(trainX.shape)
print(trainy.shape)

from tensorflow.keras.models import load_model
model = load_model('facenet_keras.h5')

model.summary()

def get_embedding(model, face_pixels):
    # PADRONIZAÇÃO
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    #TRANSFORMAR A FACE EM 1 UNICO EXEMPLO
    #(160,160) -> (1,160,160)
    samples = expand_dims(face_pixels, axis=0)

    #REALIZAR A PREDIÇÃO GERANDO O EMBEDDING
    yhat = model.predict(samples)

    return yhat[0]

newTrainX = list()

for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
newTrainX.shape

import pandas as pd
df = pd.DataFrame(data=newTrainX)
print(df)
df['target'] = trainy
print(df)
df.to_csv(('faces400.csv'))
#from sklearn.utils import shuffle
#X,y = shuffle(newTrainX, trainy, random_state=0)