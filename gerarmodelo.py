import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
#%matplotlib inline

df_desconhecidos = pd.read_csv("faces_desconhecidos.csv")
df_conhecidos = pd.read_csv("faces.csv")
df = pd.concat([df_desconhecidos, df_conhecidos])
df
X = np.array(df.drop("target", axis=1))
y = np.array(df.target)

X = np.array(df.drop("target", axis=1))
y = np.array(df.target)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

from sklearn.model_selection import train_test_split
trainX, valx, trainY, valY = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import Normalizer
norm = Normalizer(norm="l2")
tarinX = norm.transform(trainX)
valx = norm.transform(valx)

from sklearn.preprocessing import LabelEncoder
np.unique(trainY)
classes = len(np.unique(trainY))
classes

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
np.unique(trainY)

out_encoder = LabelEncoder()
out_encoder.fit(valY)
valY = out_encoder.transform(valY)
np.unique(valY)

from tensorflow.keras.utils import to_categorical
trainY = to_categorical(trainY)
valY   = to_categorical(valY)
print(valY[0])
print(trainY[0])

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
############################################################################
batch_size= 10 #8
epochs=10
############################################################################
history = model.fit(trainX, trainY,
                    epochs=epochs,
                    validation_data = (valx,valY),
                    batch_size=batch_size)

val_loss, val_acc = model.evaluate(valx, valY)
yhat_val = model.predict(valx)

valY2 = np.argmax(valY, axis = 1)
yhat_val = np.argmax(yhat_val, axis = 1)

print(valY2[0])
print(yhat_val[0])

from sklearn.metrics import confusion_matrix
def print_confusion_matrix(model_name, valY, yhat_val):
    cm = confusion_matrix(valY, yhat_val)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("Modelo: {}".format(model_name))
    print("Acuracia: {:.4f}".format(acc))
    print("Sensitividade: {:.4f}".format(sensitivity))
    print("Especificidade: {:.4f}".format(specificity))

    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5))
    #plot_confusion_matrix.show()

print_confusion_matrix("KERAS", valY2, yhat_val)

model.save("faces_ComGiro10.h5")