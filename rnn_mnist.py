import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###### data ready
mnist = tf.keras.datasets.mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

print(train_X.shape)
print(test_X.shape)

##### model
model = tf.keras.Sequential([
    #tf.keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[28,28]),
    tf.keras.layers.LSTM(units=10, return_sequences=True, input_shape=[28,28]),
    tf.keras.layers.LSTM(units=10),
    tf.keras.layers.Dense(units=10)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

##### train
history = model.fit(train_X, train_Y, epochs=25, validation_split=0.25,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.plot()
plt.legend()

plt.show()

print('evaluate')
print(model.evaluate(test_X, test_Y))