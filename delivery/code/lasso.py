import numpy as np
import pandas as pd

X_train = np.loadtxt('lasso_data/data1_input_train').T
y_train = np.loadtxt('lasso_data/data1_output_train')[:, np.newaxis]
X_val = np.loadtxt('lasso_data/data1_input_val').T
y_val = np.loadtxt('lasso_data/data1_output_val')[:, np.newaxis]

X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0))
y_train = (y_train - y_train.mean(axis=0)) / (y_train.std(axis=0))
X_val = (X_val - X_val.mean(axis=0)) / (X_val.std(axis=0))
y_val = (y_val - y_val.mean(axis=0)) / (y_val.std(axis=0))


def S(beta, gamma):
    if gamma >= beta:
        return 0
    else:
        return beta - gamma if beta > 0 else beta + gamma

def train_lasso(X, y, gamma, all_betas=False, n_iter=100, init_weights=None):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    betas = np.zeros(shape=(n_iter + 1, n_features))
    if init_weights is not None:
        beta = init_weights
    else:
        beta = np.random.randn(n_features)
    betas[0, :] = beta
    
    for _ in range(n_iter):
        for j in range(n_features):
            # Calculate y_tilde
            y_tilde = np.zeros((n_samples,))
            for i in range(n_samples):
                for k in range(n_features):
                    y_tilde[i] = X[i, k] * beta[k]
            f1 = beta[j]
            for i in range(n_samples):
                f1 += X[i, j] * (y[i] - y_tilde[i])
            beta[j] = S(f1 / n_samples, gamma)
        betas[_ + 1, :] = beta
    if all_betas:
        return betas
    else:
        return beta

import matplotlib.pyplot as plt

betas = train_lasso(X_train, y_train, gamma=0.1, all_betas=True)
fig = plt.figure(figsize=(10, 4))

for feature in range(100):
    plt.plot(np.arange(10), betas[:10, feature], alpha=0.5)
    plt.scatter(np.arange(10), betas[:10, feature], alpha=0.3)

plt.ylabel('Weights')
plt.xlabel('Number of iterations')
plt.show()


##############################################
######### Correlated Case  ###################
##############################################
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

X = np.loadtxt('lasso_data/X_cor', delimiter=',').T
y = np.loadtxt('lasso_data/y_cor', delimiter=',')[:, np.newaxis]
X = (X - X.mean(axis=0)) / (X.std(axis=0))
y = (y - y.mean(axis=0)) / (y.std(axis=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

import matplotlib.pyplot as plt

temp = np.array(betas)
for i in range(3):
    plt.plot(np.linspace(0, 1, 10), temp[:, i], label=str(i))

plt.legend()
plt.xlabel("$ \gamma $")
plt.ylabel("Weights")
plt.show()


import pandas as pd
df_X = pd.DataFrame(X_train)
df_X.corr()

plt.matshow(df_X.corr(), cmap=plt.cm.gray)
plt.colorbar()

from sklearn.linear_model import LinearRegression

l_errors = []
r_errors = []
o_errors = []
gammas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
for g in gammas:
    beta_lasso = train_lasso(X_train, y_train, gamma=g)
    l_errors.append(mean_squared_error(y_test, predict(X_test, beta=beta_lasso)))
    
    clf = Ridge(alpha=g)
    clf.fit(X_train, y_train)
    r_errors.append(mean_squared_error(y_test, clf.predict(X_test)))

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    o_errors.append(mean_squared_error(y_test, clf.predict(X_test)))


plt.plot(range(-6, 3), l_errors, label='Lasso')
plt.plot(range(-6, 3), r_errors, label='ridge')
plt.xlabel('$ \log \gamma $ ')
plt.ylabel('Mean Squared Error')
plt.legend()
