import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

'''
male = 0
female = 1

'''
df = pd.read_csv('/Users/pradys_coding_machine/python/datasets_use/titanic_train.csv')
x = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(3,)))
model.add(tf.keras.layers.Dense(9, activation='relu'))
model.add(tf.keras.layers.Dense(18, activation='relu'))
model.add(tf.keras.layers.Dense(9, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
m


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy', 'binary_accuracy'])
model.fit(x_train, y_train,
          epochs=500,
          batch_size=5)
score = model.evaluate(x_test, y_test)
print(score)



