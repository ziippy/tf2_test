import matplotlib.pyplot as plt
import numpy as np

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

plt.plot(population_inc, population_old, 'bo')
plt.show()

# outlier remove
# inc: 15.17, old:
mod_population_inc = []
mod_population_old = []
for i in range(len(population_inc)):
    if population_inc[i] > 10:
        continue
    mod_population_inc.append(population_inc[i])
    mod_population_old.append(population_old[i])

plt.plot(mod_population_inc, mod_population_old, 'bo')
plt.show()

# do linear regression

X = mod_population_inc
Y = mod_population_old
x_bar = sum(X) / len(X)
y_bar = sum(Y) / len(Y)

# lsm (least square method)
a = sum([(y-y_bar) * (x-x_bar) for y, x in list(zip(Y, X))])
a /= sum([(x-x_bar) ** 2 for x in X])
b = y_bar - a * x_bar

line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x + b

plt.plot(X, Y, 'bo')
plt.plot(line_x, line_y, 'r-')

plt.show()

