import numpy as np

num_neurons = 10
data = np.random.binomial(1, 0.5, (num_neurons, 5))
data = data * 2 - 1

print(data)