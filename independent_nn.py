import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import array
import matplotlib as plt
import tensorflow as tf
# Iris setosa = 0
# Iris versicolor = 1
# Iris virginica = 2

df = pd.read_csv('/Users/pradys_coding_machine/python/datasets_use/IRIS.csv')
# Really important, this is the how to split the data


y = df.pop('species')
x = df



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(4,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=25,
          batch_size=50)
score = model.evaluate(x_test, y_test, batch_size=50)
print(score)
Xnew = array([[5.9, 3, 5.1, 1.8]])
ynew = model.predict_classes(Xnew)

print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))