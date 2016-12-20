import numpy as np
import scipy.io


def calculate_free_energy(m, w, theta, max_energy=500):
    """
    Calculate the mean field free energy.

    :param m: Parameters of the system (n-dimensional vector).
    :param w: Weights of the system (nxn matrix).
    :param theta: The biases of the system (n-dimensional vector).
    :param max_energy: The maximum mean field free energy in the system.
    :return: The mean field free energy.
    """
    F = 0
    F -= 0.5 * np.sum(np.multiply(np.dot(w, m), m), 0)
    F -= np.dot(theta, m)
    F += np.dot(1 + m, np.log(0.5 * (1 + m)))
    F += np.dot(1 - m, np.log(0.5 * (1 - m)))
    return max(F, -max_energy)


def calculate_energy(s, w, theta, max_energy=500):
    """
    Calculate energy of a state.

    :param s: The state to calculate the energy for (n-dimensional vector).
    :param w: Weights of the system (nxn matrix).
    :param theta: The biases of the system (n-dimensional vector).
    :param max_energy: The maximum energy in the system.
    :return: The energy of the given state.
    """
    E = np.dot(theta.T, s) + 0.5 * np.sum(np.multiply(np.dot(w, s), s), 0)
    E[E > max_energy] = max_energy
    E[E < -max_energy] = -max_energy
    return E


def calculate_state_probability(s, w, theta, F, max_energy=500):
    """
    Calculate the probability that a state occurs in the system.

    :param s: The state to calculate the probability for (n-dimensional vector).
    :param w: Weights of the system (nxn matrix).
    :param theta: The biases of the system (n-dimensional vector).
    :param F: The free energy in the system.
    :param max_energy: The maximum energy in the system.
    :return: The probability that state s occurs in the system (with weights w and biases theta).
    """
    E = calculate_energy(s, w, theta, max_energy) + F
    E[E > max_energy] = max_energy
    E[E < -max_energy] = -max_energy
    return np.exp(-E)


def solve_fixed_point(f, x=0, num_iterations=100):
    """
    Solve a fixed point equation x_{t+1} = f(x_{t}).

    :param f: A method that takes x as input and has output with equal dimensionality.
    :param x: The initialization of x (x_{1}).
    :param num_iterations: The number of iterations such that x_{num_iterations} is returned.
    :return: x_{num_iterations}
    """
    for _ in range(num_iterations):
        x = f(x)
    return x


def approximate_m(m, w, theta, num_iterations=100):
    """
    Find an approximation to the following system of equations:
    m = tanh(w * m + theta)

    :param m: Initialization for the n-dimensional vector.
    :param w: The weights (nxn matrix) of the system.
    :param theta: The biases (n-dimensional vector) of the system.
    :param num_iterations: The number of iteration for solving the fixed point system.
    :return: An approximated solution for m.
    """
    return solve_fixed_point(lambda x: np.tanh(np.dot(w, m) + theta), m, num_iterations=num_iterations)


class DataLoader:
    """
    Class for loading the MatLab data.
    """

    def __init__(self, file):
        """
        Load the data from the given file.

        :param file: Path to file to load the data from (without the .mat extension).
        """
        self.data = scipy.io.loadmat(file, squeeze_me=True, struct_as_record=False)['mnist']
        self.train_images = self.transform_images(self.data.train_images)
        self.test_images = self.transform_images(self.data.test_images)
        self.train_labels = self.data.train_labels
        self.test_labels = self.data.test_labels

    def transform_images(self, data):
        """
        Convert a (m x n x p) array to a (p x m x n) array and apply some additional transformations.

        :param data: Data to transform.
        :return: Transformed data.
        """
        reshaped = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        swapped_axes = np.swapaxes(reshaped, 0, 1)
        return (swapped_axes > 122) * 2 - 1


class BoltzmannMFT:
    """
    The Boltzmann system using the Mean Field Theory to approximate the normalizing constant.
    """

    def __init__(self, training_samples, learning_rate=0.001):
        """
        Initialize the Boltzmann system.

        :param training_samples: The training samples (nxm matrix where n is the number of samples and m is the dimensionality of the samples).
        """
        # Set a seed
        np.random.seed(0)

        # Store the learning rate
        self.learning_rate = learning_rate

        # Save the training samples
        self.training_samples = training_samples

        # Store the number of training samples
        self.num_samples = training_samples.shape[0]

        # Number of neurons to use in the Boltzmann machine (n = num_neurons)
        self.num_neurons = training_samples.shape[1]

        # The initial weights in the system (nxn matrix)
        self.w = 2 * np.random.uniform(-1, 1, (self.num_neurons, self.num_neurons))

        # Remove self-loops in the system
        np.fill_diagonal(self.w, 0)

        # The initial biases in the system (n-dimensional vector)
        self.theta = np.random.uniform(-1, 1, (self.num_neurons, 1)).ravel()

        # The initial parameters in the system (n-dimensional vector)
        self.m = np.random.uniform(-1, 1, (self.num_neurons, 1)).ravel()

    def train_iteration(self, factor=0.99):
        """
        Perform one training iteration.

        :param factor: A correction factor to avoid log(0) which can happen when factor=1.
        """
        # Find an approximation for the parameters
        self.m = approximate_m(self.m, self.w, self.theta) * factor

        # Approximate the normalizing constant
        self.F = calculate_free_energy(self.m, self.w, self.theta)

        training_samples_squared = np.dot(self.training_samples.T, self.training_samples)
        stat_1_c = np.squeeze(np.asarray(np.sum(self.training_samples, 0) / float(self.num_samples)))
        stat_2_c = training_samples_squared / float(self.num_samples)

        p = calculate_state_probability(self.training_samples.T, self.w, self.theta, self.F)
        stat_1 = np.dot(self.training_samples.T, p)
        stat_2 = np.dot(training_samples_squared, np.sum(p))

        dLdtheta = stat_1_c - stat_1
        dLdw = stat_2_c - stat_2
        self.w -= self.learning_rate * dLdw
        self.theta -= self.learning_rate * dLdtheta


data = DataLoader('mnistAll')
training_samples_mask = (data.train_labels == 8)
training_samples = data.train_images[training_samples_mask]
training_samples = training_samples[list(range(1, 100))]

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
boltzmann = BoltzmannMFT(training_samples)
for _ in range(3):
    print(_)
    boltzmann.train_iteration()
ax1.imshow(boltzmann.theta.reshape(28, 28))
ax2.imshow(boltzmann.w)
plt.show()