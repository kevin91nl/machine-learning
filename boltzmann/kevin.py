import random

import scipy.io
import numpy as np
from scipy.stats import bernoulli





# data = DataLoader('mnistAll.mat')
# print(data.train_images.shape)

# Define the number of neurons
num_neurons = 784

# Initialize the weights and biases
w = 2 * np.random.normal(0, 1, (num_neurons, num_neurons))
theta = np.random.normal(0, 1, (num_neurons, 1)).ravel()
m = np.random.normal(0, 1, (num_neurons, 1)).ravel()


def approximate_m(m, w, theta, num_iterations=10):
    for _ in range(num_iterations):
        m = np.tanh(np.dot(w, m) - theta)
    return m


def calculate_log_qi(s, m, epsilon=0.00001):
    return s.shape[0] * np.log(1 / 2) + np.log(s * m + 1 + epsilon)


def calculate_energy(s, w, theta):
    energy = np.dot(theta, s)
    for i in range(s.shape[0]):
        for j in range(s.shape[0]):
            energy += 0.5 * w[i, j] * s[i] * s[j]
    return energy


m = approximate_m(m, w, theta)
s = np.ones((num_neurons,))
qi = np.exp(calculate_log_qi(s, m))
num_samples = 10
samples = [bernoulli.rvs(0.5, size=(num_neurons,)) * 2 - 1 for _ in range(num_samples)]
for s in samples:

log_qis = [calculate_log_qi(s, m) for s in samples]
print(log_qis[0])
s_q = -1 * np.sum([np.exp(log_qi) * log_qi for log_qi in log_qis])
expected_energy = [calculate_energy(s, w, theta) * np.exp(log_qi) for log_qi, s in zip(log_qis, samples)]
print(expected_energy)