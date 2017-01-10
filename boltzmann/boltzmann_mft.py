import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix


def calculate_free_energy(m, w, theta):
    """
    Calculate the mean field free energy.

    :param m: Parameters of the system (n-dimensional vector).
    :param w: Weights of the system (nxn matrix).
    :param theta: The biases of the system (n-dimensional vector).
    :return: The mean field free energy.
    """
    F = 0
    # F -= 0.5 * np.sum(np.multiply(np.dot(w, np.asmatrix(m).T), np.asmatrix(m).T), 0)
    # F -= np.dot(np.asmatrix(theta), np.asmatrix(m).T)
    # F += 0.5 * np.sum(np.multiply(1 + m, np.log(0.5 * (1 + m))))
    # F += 0.5 * np.sum(np.multiply(1 - m, np.log(0.5 * (1 - m))))
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            F -= 0.5 * w[i, j] * m[i] * m[j]
        F -= theta[0, i] * m[i]
        F += 0.5 * (1 + m[i]) * np.log(0.5 * (1 + m[i]))
        F += 0.5 * (1 - m[i]) * np.log(0.5 * (1 - m[i]))
    return F


def calculate_energy(s, w, theta):
    """
    Calculate energy of a state.

    :param s: The states to calculate the probability for (mxn matrix where m is the number of states).
    :param w: Weights of the system (nxn matrix).
    :param theta: The biases of the system (n-dimensional vector).
    :return: The energy of the given states.
    """
    E = np.dot(np.matrix(theta), s.T) + 0.5 * np.sum(np.multiply(np.dot(w, s.T), s.T), 0)
    return np.squeeze(np.asarray(E))


def calculate_state_probability(s, w, theta, F):
    """
    Calculate the probability that a state occurs in the system.

    :param s: The states to calculate the probability for (mxn matrix where m is the number of states).
    :param w: Weights of the system (nxn matrix).
    :param theta: The biases of the system (n-dimensional vector).
    :param F: The free energy in the system.
    :return: The probability that state s occurs in the system (with weights w and biases theta).
    """
    E = calculate_energy(s, w, theta)
    U = F + E
    return np.squeeze(np.asarray(np.exp(U)))


def solve_fixed_point(f, x=0, num_iterations=10):
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
        :param flip_rate: Rate for flipping bits randomly, 0 means no flipping and 1 means all flipped and the
                          maximum entropy is obtained for flip_rate = 0.5.
        """
        self.data = scipy.io.loadmat(file, squeeze_me=True, struct_as_record=False)['mnist']
        self.train_images = self.data.train_images
        self.test_images = self.data.test_images
        self.train_labels = self.data.train_labels
        self.test_labels = self.data.test_labels

    def load_images(self, flip_rate=0):
        self.train_images = self.transform_images(self.train_images)
        self.test_images = self.transform_images(self.test_images)
        if flip_rate > 0:
            noise = np.random.binomial(1, flip_rate, self.train_images.shape) * -2 + 1
            self.train_images = np.multiply(self.train_images, noise)

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

    def __init__(self, num_neurons):
        """
        Initialize the Boltzmann system.
        """
        # Number of neurons to use in the Boltzmann machine (n = num_neurons)
        self.num_neurons = num_neurons

        # The initial weights in the system (nxn matrix)
        self.w = 2 * np.random.uniform(-1, 1, (self.num_neurons, self.num_neurons))

        # Remove self-loops in the system
        np.fill_diagonal(self.w, 0)

        # The initial biases in the system (n-dimensional vector)
        self.theta = np.random.uniform(-1, 1, (self.num_neurons, 1)).ravel()

        # The initial parameters in the system (n-dimensional vector)
        self.m = np.random.uniform(-1, 1, (self.num_neurons, 1)).ravel()

    def train(self, training_samples):
        """
        Perform one training iteration.

        :param training_samples: The training samples (nxm matrix where n is the number of samples and m is the dimensionality of the samples).
        :param factor: A correction factor to avoid log(0) which can happen when factor=1.
        """
        s_mean_clamped = np.squeeze(np.asarray(np.mean(training_samples, 0)))
        s_cov_clamped = np.cov(training_samples.T)
        self.m = s_mean_clamped
        C = s_cov_clamped - np.dot(np.asmatrix(s_mean_clamped).T, np.asmatrix(s_mean_clamped))
        delta = np.zeros(s_cov_clamped.shape)
        m_squared = np.multiply(self.m, self.m)
        np.fill_diagonal(delta, 1. / (1. - m_squared))
        self.w = delta - np.linalg.inv(C)
        self.theta = np.arctanh(self.m) - np.dot(self.w, self.m)
        self.F = calculate_free_energy(self.m, self.w, self.theta)

    def save(self, file):
        """
        Save the parameters to a file.
        
        :param file: Path to the output file. 
        """
        with open(file, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(file):
        """
        Load the parameters from a file.

        :param file: Path to the input file.
        :return BoltzmannMTF object
        """
        with open(file, 'rb') as handle:
            obj = pickle.load(handle)
        return obj


# Train (or load when exist) the Boltzmann machines
boltzmann = []
for digit in range(0, 10):
    file = 'digit_%d.pickle' % digit
    boltzmann.append(BoltzmannMFT(28 * 28))
    if not os.path.exists(file):
        print('Training Boltzmann machine for digit %d and store the result to %s...' % (digit, file))
        data = DataLoader('mnistAll')
        data.load_images(0.1)
        training_samples = data.train_images[data.train_labels == digit]
        boltzmann[digit].train(training_samples)
        boltzmann[digit].save(file)
    else:
        print('Loading Boltzmann machine for digit %d from file %s...' % (digit, file))
        boltzmann[digit] = BoltzmannMFT.load(file)

file = 'probabilities.pickle'
if not os.path.exists(file):
    data = DataLoader('mnistAll')
    data.load_images(0.1)
    testset = data.test_images[data.test_labels]
    probabilities = []
    for digit in range(0, 10):
        print('Calculating classifier probabilities for the Boltzmann machine for digit %d...' % digit)
        clf_probabilities = calculate_state_probability(testset, boltzmann[digit].w, boltzmann[digit].theta, boltzmann[digit].F)
        probabilities.append(clf_probabilities)
    probabilities = np.matrix(probabilities)
    with open(file, 'wb') as handle:
        pickle.dump(probabilities, handle)
else:
    print('Loading classifier probabilities...')
    with open(file, 'rb') as handle:
        probabilities = pickle.load(handle)

data = DataLoader('mnistAll')
real = data.test_labels
predicted = np.squeeze(np.asarray(np.argmax(probabilities, 0)))
print(predicted[:20])
print(real[:20])
cm = confusion_matrix(real, predicted)
plt.imshow(cm)
plt.show()

for digit in range(0, 10):
    print(np.min(boltzmann[digit].m))
    print(np.max(boltzmann[digit].m))