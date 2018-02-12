import numpy as np
import matplotlib.pyplot as plt
from utils_2nd import *


#Define the Forward function.
def forward(X, W1, b1, W2, b2):
    hidden = X.dot(W1) + b1
    hidden[hidden < 0] = 0
    return hidden, softmax(hidden.dot(W2) + b2)


#Define the derivative functions.
def derivative_W2(hidden, output, T):
    return hidden.T.dot(output - T)

def derivative_b2(output, T):
    return (output - T).sum(axis = 0)

def derivative_W1(W2, output, T, X, hidden):
    dZ = (output - T).dot(W2.T) * (hidden > 0)
    return X.T.dot(dZ)

def derivative_b1(W2, output, T, hidden):
    dZ = (output - T).dot(W2.T) * (hidden > 0)
    return dZ.sum(axis = 0)

def train_normal(X, Y, T, learning_rate, epochs,
                batch_size, Number_hidden, print_batch):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    batches = len(Xtrain) // batch_size

    #Get the shape parameters.
    N, D = X.shape
    K = len(set(Y))
    M = Number_hidden

    #Initialize the weights and the regularization parameter.
    W1, b1, W2, b2 = initiate_weights_2layer(N, D, M, K)
    reg = 0.01

    #Initialize the costs function
    costs = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            Hbatch, Obatch = forward(Xbatch, W1, b1, W2, b2)

            W2 -= learning_rate * (derivative_W2(Hbatch, Obatch, Tbatch) + reg * W2)
            b2 -= learning_rate * (derivative_b2(Obatch, Tbatch) + reg * b2)
            W1 -= learning_rate * (derivative_W1(W2, Obatch, Tbatch, Xbatch, Hbatch) + reg * W1)
            b1 -= learning_rate * (derivative_b1(W2, Obatch, Tbatch, Hbatch) + reg * b1)


            if batch % print_batch == 0:
                _, Otest = forward(Xtest, W1, b1, W2, b2)
                c = cross_entropy(Ttest, Otest)
                costs.append(c)
                acc = check_acc(Ytest, final_result(Otest))
                print("Epoch", epoch, "Batch", batch, "Cost", c, "Acc", acc)


    return costs


def train_momentum(X, Y, T, learning_rate, epochs,
                batch_size, Number_hidden,
                print_batch, mu_momentum):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    batches = len(Xtrain) // batch_size

    #Get the shape parameters.
    N, D = X.shape
    K = len(set(Y))
    M = Number_hidden

    #Initialize the weights and the regularization parameter.
    W1, b1, W2, b2 = initiate_weights_2layer(N, D, M, K)
    reg = 0

    #Initialize the parameters for Momentum.
    mu = mu_momentum
    dW1, db1, dW2, db2 = 0, 0, 0, 0

    #Initialize the costs function
    costs = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            Hbatch, Obatch = forward(Xbatch, W1, b1, W2, b2)

            dW2 = mu*dW2 - learning_rate * (derivative_W2(Hbatch, Obatch, Tbatch) + reg * W2)
            W2 += dW2

            db2 = mu*db2 - learning_rate * (derivative_b2(Obatch, Tbatch) + reg * b2)
            b2 += db2

            dW1 = mu*dW1 - learning_rate * (derivative_W1(W2, Obatch, Tbatch, Xbatch, Hbatch) + reg * W1)
            W1 += dW1

            db1 = mu*db1 - learning_rate * (derivative_b1(W2, Obatch, Tbatch, Hbatch) + reg * b1)
            b1 += db1

            if batch % print_batch == 0:
                _, Otest = forward(Xtest, W1, b1, W2, b2)
                c = cross_entropy(Ttest, Otest)
                costs.append(c)
                acc = check_acc(Ytest, final_result(Otest))
                print("Epoch", epoch, "Batch", batch, "Cost", c, "Acc", acc)


    return costs

def train_adagrad(X, Y, T, learning_rate, epochs,
                batch_size, Number_hidden,
                print_batch, starting_cache):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    batches = len(Xtrain) // batch_size

    #Get the shape parameters.
    N, D = X.shape
    K = len(set(Y))
    M = Number_hidden

    #Initialize the weights and the regularization parameter.
    W1, b1, W2, b2 = initiate_weights_2layer(N, D, M, K)
    reg = 0.01

    #Initialize the cash parameters.
    cacheW2, cacheb2, cacheW1, cacheb1 = starting_cache, starting_cache, starting_cache, starting_cache
    esp = 10e-8

    #Initialize the costs function
    costs = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            Hbatch, Obatch = forward(Xbatch, W1, b1, W2, b2)

            gW2 = (derivative_W2(Hbatch, Obatch, Tbatch) + reg * W2)
            cacheW2 += gW2 * gW2
            W2 -= learning_rate * gW2 / (np.sqrt(cacheW2 + esp))

            gb2 = (derivative_b2(Obatch, Tbatch) + reg * b2)
            cacheb2 += gb2 * gb2
            b2 -= learning_rate * gb2 / np.sqrt(cacheb2 + esp)

            gW1 = (derivative_W1(W2, Obatch, Tbatch, Xbatch, Hbatch) + reg * W1)
            cacheW1 += gW1 * gW1
            W1 -= learning_rate * gW1 / np.sqrt(cacheW1 + esp)

            gb1 = (derivative_b1(W2, Obatch, Tbatch, Hbatch) + reg * b1)
            cacheb1 += gb1 * gb1
            b1 -= learning_rate * gb1 / np.sqrt(cacheb1 + esp)

            if batch % print_batch == 0:
                _, Otest = forward(Xtest, W1, b1, W2, b2)
                c = cross_entropy(Ttest, Otest)
                costs.append(c)
                acc = check_acc(Ytest, final_result(Otest))
                print("Epoch", epoch, "Batch", batch, "Cost", c, "Acc", acc)


    return costs



def train_rmsprop(X, Y, T, learning_rate, epochs,
                batch_size, Number_hidden,
                print_batch, starting_cache, decay_rate):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    batches = len(Xtrain) // batch_size

    #Get the shape parameters.
    N, D = X.shape
    K = len(set(Y))
    M = Number_hidden

    #Initialize the weights and the regularization parameter.
    W1, b1, W2, b2 = initiate_weights_2layer(N, D, M, K)
    reg = 0.01

    #Initialize the cash parameters.
    cacheW2, cacheb2, cacheW1, cacheb1 = starting_cache, starting_cache, starting_cache, starting_cache
    esp = 10e-10
    decay_rate = decay_rate

    #Initialize the costs function
    costs = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            Hbatch, Obatch = forward(Xbatch, W1, b1, W2, b2)

            gW2 = (derivative_W2(Hbatch, Obatch, Tbatch) + reg * W2)
            cacheW2 = decay_rate * cacheW2 + (1 - decay_rate) * gW2 * gW2
            W2 -= learning_rate * gW2 / (np.sqrt(cacheW2 + esp))

            gb2 = (derivative_b2(Obatch, Tbatch) + reg * b2)
            cacheb2 = decay_rate * cacheb2 + (1 - decay_rate) * gb2 * gb2
            b2 -= learning_rate * gb2 / np.sqrt(cacheb2 + esp)

            gW1 = (derivative_W1(W2, Obatch, Tbatch, Xbatch, Hbatch) + reg * W1)
            cacheW1 = decay_rate * cacheW1 + (1 - decay_rate) * gW1 * gW1
            W1 -= learning_rate * gW1 / np.sqrt(cacheW1 + esp)

            gb1 = (derivative_b1(W2, Obatch, Tbatch, Hbatch) + reg * b1)
            cacheb1 = decay_rate * cacheb1 + (1 - decay_rate) * gb1 * gb1
            b1 -= learning_rate * gb1 / np.sqrt(cacheb1 + esp)

            if batch % print_batch == 0:
                _, Otest = forward(Xtest, W1, b1, W2, b2)
                c = cross_entropy(Ttest, Otest)
                costs.append(c)
                acc = check_acc(Ytest, final_result(Otest))
                print("Epoch", epoch, "Batch", batch, "Cost", c, "Acc", acc)


    return costs



def train_rmsprop_m(X, Y, T, learning_rate, epochs,
                batch_size, Number_hidden,
                print_batch, starting_cache, decay_rate, mu_momentum):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    batches = len(Xtrain) // batch_size

    #Get the shape parameters.
    N, D = X.shape
    K = len(set(Y))
    M = Number_hidden

    #Initialize the weights and the regularization parameter.
    W1, b1, W2, b2 = initiate_weights_2layer(N, D, M, K)
    reg = 0.01

    #Initialize the cache parameters.
    cacheW2, cacheb2, cacheW1, cacheb1 = starting_cache, starting_cache, starting_cache, starting_cache
    esp = 1e-8
    decay_rate = decay_rate

    #Initialize the Momentum parameters.
    mu = mu_momentum
    dW1, db1, dW2, db2 = 0, 0, 0, 0

    #Initialize the costs function
    costs = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            Hbatch, Obatch = forward(Xbatch, W1, b1, W2, b2)

            gW2 = (derivative_W2(Hbatch, Obatch, Tbatch) + reg * W2)
            gb2 = (derivative_b2(Obatch, Tbatch) + reg * b2)
            gW1 = (derivative_W1(W2, Obatch, Tbatch, Xbatch, Hbatch) + reg * W1)
            gb1 = (derivative_b1(W2, Obatch, Tbatch, Hbatch) + reg * b1)

            cacheW2 = decay_rate * cacheW2 + (1 - decay_rate) * gW2 * gW2
            cacheb2 = decay_rate * cacheb2 + (1 - decay_rate) * gb2 * gb2
            cacheW1 = decay_rate * cacheW1 + (1 - decay_rate) * gW1 * gW1
            cacheb1 = decay_rate * cacheb1 + (1 - decay_rate) * gb1 * gb1

            dW2 = mu * dW2 - (1 - mu) * learning_rate * gW2 / np.sqrt(cacheW2 + esp)
            db2 = mu * db2 - (1 - mu) * learning_rate * gb2 / np.sqrt(cacheb2 + esp)
            dW1 = mu * dW1 - (1 - mu) * learning_rate * gW1 / np.sqrt(cacheW1 + esp)
            db1 = mu * db1 - (1 - mu) * learning_rate * gb1 / np.sqrt(cacheb1 + esp)

            W2 += dW2
            b2 += db2
            W1 += dW1
            b1 += db1



            if batch % print_batch == 0:
                _, Otest = forward(Xtest, W1, b1, W2, b2)
                c = cross_entropy(Ttest, Otest)
                costs.append(c)
                acc = check_acc(Ytest, final_result(Otest))
                print("Epoch", epoch, "Batch", batch, "Cost", c, "Acc", acc)


    return costs


def train_adam(X, Y, T, learning_rate, epochs,
                batch_size, Number_hidden,
                print_batch, starting_cache, decay_rate, mu_momentum):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    batches = len(Xtrain) // batch_size

    #Get the shape parameters.
    N, D = X.shape
    K = len(set(Y))
    M = Number_hidden

    #Initialize the weights and the regularization parameter.
    W1, b1, W2, b2 = initiate_weights_2layer(N, D, M, K)
    reg = 0.01

    #Initialize the cache parameters.
    cacheW2, cacheb2, cacheW1, cacheb1 = starting_cache, starting_cache, starting_cache, starting_cache
    esp = 1e-8
    decay_rate_cache = decay_rate

    #Initialize the Momentum parameters.
    mu_decay = mu_momentum
    dW1, db1, dW2, db2 = 0, 0, 0, 0

    #Initialize the costs function
    costs = []

    t = 1

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            Hbatch, Obatch = forward(Xbatch, W1, b1, W2, b2)

            #Get the values of the gradients.
            gW2 = (derivative_W2(Hbatch, Obatch, Tbatch) + reg * W2)
            gb2 = (derivative_b2(Obatch, Tbatch) + reg * b2)
            gW1 = (derivative_W1(W2, Obatch, Tbatch, Xbatch, Hbatch) + reg * W1)
            gb1 = (derivative_b1(W2, Obatch, Tbatch, Hbatch) + reg * b1)

            #Get the values of the caches.
            cacheW2 = decay_rate_cache * cacheW2 + (1 - decay_rate_cache) * gW2 * gW2
            cacheb2 = decay_rate_cache * cacheb2 + (1 - decay_rate_cache) * gb2 * gb2
            cacheW1 = decay_rate_cache * cacheW1 + (1 - decay_rate_cache) * gW1 * gW1
            cacheb1 = decay_rate_cache * cacheb1 + (1 - decay_rate_cache) * gb1 * gb1

            #Get the values for the momentums
            dW2 = mu_decay * dW2 + (1 - mu_decay) * gW2
            db2 = mu_decay * db2 + (1 - mu_decay) * gb2
            dW1 = mu_decay * dW1 + (1 - mu_decay) * gW1
            db1 = mu_decay * db1 + (1 - mu_decay) * gb1

            #correction for the caches
            cacheW2_hat = cacheW2 / (1 - np.power(decay_rate_cache,t))
            cacheb2_hat = cacheb2 / (1 - np.power(decay_rate_cache,t))
            cacheW1_hat = cacheW1 / (1 - decay_rate_cache ** t)
            cacheb1_hat = cacheb1 / (1 - np.power(decay_rate_cache,t))

            #correction for the momentums
            dW2_hat = dW2 / (1 - np.power(mu_decay,t))
            db2_hat = db2 / (1 - np.power(mu_decay,t))
            dW1_hat = dW1 / (1 - np.power(mu_decay,t))
            db1_hat = db1 / (1 - np.power(mu_decay,t))

            #Update the final weights
            W2 -= learning_rate * dW2_hat / np.sqrt(cacheW2_hat + esp)
            b2 -= learning_rate * db2_hat / np.sqrt(cacheb2_hat + esp)
            W1 -= learning_rate * dW1_hat / np.sqrt(cacheW1_hat + esp)
            b1 -= learning_rate * db1_hat / np.sqrt(cacheb1_hat + esp)

            #Update time.
            t += 1

            if batch % print_batch == 0:
                _, Otest = forward(Xtest, W1, b1, W2, b2)
                # print(cacheW1)
                c = cross_entropy(Ttest, Otest)
                costs.append(c)
                acc = check_acc(Ytest, final_result(Otest))
                print("Epoch", epoch, "Batch", batch, "Cost", c, "Acc", acc)


    return costs



def main(Normal = True, Mom = True, Adagrad = True, rms = True,
        rms_m = True, Adam = True):

    X, Y, _, _ = get_data(PCA_data = False, num_parameters = 300)
    T = onehotencode(Y)

    print_batch = 10
    learning_rate = 10e-5
    epochs = 5

    if Normal:
        costs_normal = train_normal(X, Y, T, learning_rate = learning_rate,
                                epochs = epochs, batch_size = 500,
                                Number_hidden = 300, print_batch = print_batch)
        x1 = np.linspace(0, 1, num=len(costs_normal))
        _ = plt.plot(x1, costs_normal, label = "normal")

    if Mom:
        costs_momentum = train_momentum(X, Y, T, learning_rate = learning_rate,
                                epochs = epochs, batch_size = 500,
                                Number_hidden = 300, print_batch = print_batch,
                                mu_momentum = 0.8)
        _ = plt.plot(x1, costs_momentum, label = "momentum = 0.9")

    if Adagrad:
        costs_adagrad = train_adagrad(X, Y, T, learning_rate = learning_rate * 10,
                                epochs = epochs, batch_size = 500,
                                Number_hidden = 300, print_batch = print_batch,
                                starting_cache = 0)
        _ = plt.plot(x1, costs_adagrad, label = "adagrad")

    if rms:
        costs_rmsprop = train_rmsprop(X, Y, T, learning_rate = learning_rate * 10,
                                epochs = epochs, batch_size = 500,
                                Number_hidden = 300, print_batch = print_batch,
                                starting_cache = 0, decay_rate = 0.999)
        _ = plt.plot(x1, costs_rmsprop, label = "Rmsprop c = 0")

    if rms_m:
        costs_rmsprop_m = train_rmsprop_m(X, Y, T,
                                learning_rate = learning_rate * 10,
                                epochs = epochs, batch_size = 500,
                                Number_hidden = 300, print_batch = print_batch,
                                starting_cache = 1, decay_rate = 0.999,
                                mu_momentum = 0.9)
        _ = plt.plot(x1, costs_rmsprop_m, label = "Rmsprop + m c = 1")

    if Adam:
        costs_adam = train_adam(X, Y, T, learning_rate = learning_rate * 10,
                                epochs = epochs, batch_size = 500,
                                Number_hidden = 300, print_batch = print_batch,
                                starting_cache = 0, decay_rate = 0.999,
                                mu_momentum = 0.9)
        _ = plt.plot(x1, costs_adam, label = "Adam")


    _ = plt.legend()
    plt.show()


if __name__ == "__main__":
    main(Normal = True, Mom = True,
        Adagrad = True, rms = False,
        rms_m = False, Adam = True)
