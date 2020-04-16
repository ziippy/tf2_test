import tensorflow as tf
import numpy as np

###### data ready
X = []
Y = []

for i in range(6):
    # [0,1,2,3], [1,2,3,4], ...
    lst = list(range(i,i+4))

    # divide 10 for lst, and save to X
    X.append(list(map(lambda c: [c/10], lst)))
    print('X in for:', X)

    # save correct answer to Y
    Y.append((i+4)/10)
    print('Y in for:', Y)

X = np.array(X)
Y = np.array(Y)
for i in range(len(X)):
    print('X:', X[i], 'Y:', Y[i])

##### model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[4,1]),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam',
              loss='mse')

model.summary()

##### train
model.fit(X, Y, epochs=100, verbose=0)
print(model.predict(X))

##### test
print(model.predict(np.array([[[0.6],[0.7],[0.8],[0.9]]])))
print(model.predict(np.array([[[-0.1],[0.0],[0.1],[0.2]]])))