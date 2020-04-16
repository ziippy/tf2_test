import matplotlib.pyplot as plt
import tensorflow as tf

x = range(20)
y = tf.random.normal([20],0,1)

plt.plot(x,y)
plt.show()

plt.plot(x,y, 'bo')
plt.show()

#========================
random_normal = tf.random.uniform([100000], 0, 1)
plt.hist(random_normal, bins=100)
plt.show()

#========================
random_normal = tf.random.normal([100000], 0, 1)
plt.hist(random_normal, bins=100)
plt.show()