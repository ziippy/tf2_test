import math
import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[1], [1], [1], [0]])
w = tf.random.normal([2],0,1)
b = tf.random.normal([1],0,1)
b_x = 1

for i in range(2000):
    error_sum = 0

    for j in range(4):
        output = sigmoid(np.sum(x[j]*w) + b_x*b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error
        
    if i % 200 == 199:
        print(i, error_sum)

# evaluation
for k in range(4):
    print('X:', x[k], 'Y:', y[k], 'Output:', sigmoid(np.sum(x[k]*w)+b))