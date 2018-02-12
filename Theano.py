import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as Te
import theano
import utils_2 as U
from datetime import datetime

def accuracy(Y, pY):
    return np.mean(Y == pY)

def train_theano(X, Y, T, Hidden_1, learning_rate, reg, epochs, batch_size):


    #Create the test and training sets.
    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    #We will create the 1 layer network.
    #Define all the parameters here.
    N, D = X.shape
    H1 = Hidden_1
    K = len(set(Y))
    batches = len(Ttrain) // batch_size

    #Define the weights for those parameters.
    W1_init = np.random.randn(D, H1) / np.sqrt(D)
    b1_init = np.zeros(H1)
    W2_init = np.random.randn(H1, K) / np.sqrt(H1)
    b2_init = np.zeros(K)

    #Inititalize the theano varialbles.
    thX = Te.matrix('X')
    thT = Te.matrix("T")

    #Put the value of these weights to the theano.shared to be updated by grad
    W1 = theano.shared(W1_init, "W1")
    b1 = theano.shared(b1_init, "b1")
    W2 = theano.shared(W2_init, "W2")
    b2 = theano.shared(b2_init, "b2")

    #Get the hidden and output equations.
    hidden = Te.nnet.relu(thX.dot(W1) + b1)
    thpY = Te.nnet.softmax(hidden.dot(W2) + b2)

    #Now define the cost function
    cost_func = - ((thT * Te.log(thpY)).sum()) \
                + reg * (W1 * W1).sum() \
                + reg * (b1 * b1).sum() \
                + reg * (W2 * W2).sum() \
                + reg * (b2 * b2).sum()
    prediction = Te.argmax(thpY, axis = 1)

    #Now we will define the update values for all the variables that we used.
    W1_update = W1 - learning_rate * Te.grad(cost_func, W1)
    b1_update = b1 - learning_rate * Te.grad(cost_func, b1)
    W2_update = W2 - learning_rate * Te.grad(cost_func, W2)
    b2_update = b2 - learning_rate * Te.grad(cost_func, b2)

    train = theano.function(
            inputs = [thX, thT],
            updates = [(W1, W1_update), (b1, b1_update),
            (W2, W2_update), (b2, b2_update)]
    )

    get_prediction = theano.function(
            inputs = [thX, thT],
            outputs= [cost_func, prediction]
    )

    costs_test = []
    time = datetime.now()
    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            train(Xbatch, Tbatch)

            if batch % 10 == 0:
                cost_val, prediction = get_prediction(Xtest, Ttest)
                acc = accuracy(Ytest, prediction)
                costs_test.append(cost_val)
                print("Epoch", epoch, "Cost", cost_val, "Acc", acc)


    print(datetime.now() - time)
    return costs_test

def train_momentum(X, Y, T, Hidden_1,
                    learning_rate, reg, epochs,
                    batch_size, mu_momentum):

    #Create the test and training sets.
    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    #We will create the 1 layer network.
    #Define all the parameters here.
    N, D = X.shape
    H1 = Hidden_1
    K = len(set(Y))
    batches = len(Ttrain) // batch_size

    #Inititalize the weights.
    W1_init = np.random.randn(D, H1) / np.sqrt(D)
    b1_init = np.zeros(H1)
    W2_init = np.random.randn(H1, K) / np.sqrt(H1)
    b2_init = np.zeros(K)


    #Assign the theano placeholders.
    #Add the weights to the theano.shared
    thX = Te.matrix("X")
    thT = Te.matrix("T")
    W1 = theano.shared(W1_init, name="W1")
    b1 = theano.shared(b1_init, name="b1")
    W2 = theano.shared(W2_init, name="W2")
    b2 = theano.shared(b2_init, name="b2")

    #Initialize the momentum parameters.
    dW1_init = np.zeros([D,H1])
    db1_init = np.zeros(H1)
    dW2_init = np.zeros([H1, K])
    db2_init = np.zeros(K)
    dW1 = theano.shared(dW1_init, name="dW1")
    db1 = theano.shared(db1_init, name="db1")
    dW2 = theano.shared(dW2_init, name="dW2")
    db2 = theano.shared(db2_init, name="db2")


    #Define our Neural network
    hidden = Te.nnet.relu(thX.dot(W1) + b1)
    thpY = Te.nnet.softmax(hidden.dot(W2) + b2)

    #Define the cost fucntion and the prediction function.
    cost_func = -(thT * Te.log(thpY)).sum() \
                + reg * (W1 * W1).sum() \
                + reg * (b1 * b1).sum() \
                + reg * (W2 * W2).sum() \
                + reg * (b2 * b2).sum()
    prediction = Te.argmax(thpY, axis = 1)

    #Update the weights with momentum.
    dW2_update = - learning_rate * Te.grad(cost_func, W2) + mu_momentum * dW2
    W2_update = W2 + mu_momentum * dW2 - learning_rate * Te.grad(cost_func, W2)
    db2_update = - learning_rate * Te.grad(cost_func, b2) + mu_momentum * db2
    b2_update = b2 - learning_rate * Te.grad(cost_func, b2) + mu_momentum * db2
    dW1_update = - learning_rate * Te.grad(cost_func, W1) + mu_momentum * dW1
    W1_update = W1 - learning_rate * Te.grad(cost_func, W1) + mu_momentum * dW1
    db1_update = - learning_rate * Te.grad(cost_func, b1) + mu_momentum * db1
    b1_update = b1 - learning_rate * Te.grad(cost_func, b1) + mu_momentum * db1

    #Define the train and predict function.
    train = theano.function(
        inputs=[thX, thT],
        updates=[(W1, W1_update), (b1, b1_update),
                (W2, W2_update), (b2, b2_update)] +
                [(dW1, dW1_update), (dW2, dW2_update),
                (db2, db2_update), (db1, db1_update)]
    )

    get_prediction = theano.function(
            inputs=[thX, thT],
            outputs=[cost_func, prediction]
    )

    costs_test = []
    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            train(Xbatch, Tbatch)
            # train2(Xbatch, Tbatch)

            if batch % 10 == 0:
                cost_val, pred = get_prediction(Xtest, Ttest)
                acc = accuracy(Ytest, pred)
                costs_test.append(cost_val)
                print("Epoch", epoch, "Cost", cost_val, "Acc", acc)


    return costs_test

def train_Adagrad(X, Y, T, Hidden_1,
                    learning_rate, reg, epochs,
                    batch_size, cache):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    #We will create the 1 layer network.
    #Define all the parameters here.
    N, D = X.shape
    H1 = Hidden_1
    K = len(set(Y))
    batches = len(Ttrain) // batch_size

    #Inititalize the weights.
    W1_init = np.random.randn(D, H1) / np.sqrt(D)
    b1_init = np.zeros(H1)
    W2_init = np.random.randn(H1, K) / np.sqrt(H1)
    b2_init = np.zeros(K)

    #Setup the X and Y placeholders along with weight variables.
    thX = Te.matrix("X")
    thT = Te.matrix("T")
    W1 = theano.shared(W1_init, "W1")
    b1 = theano.shared(b1_init, "b1")
    W2 = theano.shared(W2_init, "W2")
    b2 = theano.shared(b2_init, "b2")

    #Setup the cache parameters for the different weights.
    if cache == 1:
        cacheW1_init = np.ones([D, H1])
        cacheb1_init = np.ones(H1)
        cacheW2_init = np.ones([H1, K])
        cacheb2_init = np.ones(K)
    else:
        cacheW1_init = np.zeros([D, H1])
        cacheb1_init = np.zeros(H1)
        cacheW2_init = np.zeros([H1, K])
        cacheb2_init = np.zeros(K)

    #Setup the shared variables for the cache weights.
    cacheW1 = theano.shared(cacheW1_init, "cacheW1")
    cacheb1 = theano.shared(cacheb1_init, "cacheb1")
    cacheW2 = theano.shared(cacheW2_init, "cacheW2")
    cacheb2 = theano.shared(cacheb2_init, "cacheb2")

    #Setup the model.
    hidden = Te.nnet.relu(thX.dot(W1) + b1)
    thpY = Te.nnet.softmax(hidden.dot(W2) + b2)

    #Define the cost function and the prediction.
    cost_func = -(thT * Te.log(thpY)).sum() \
                + reg * (W1 * W1).sum() \
                + reg * (b1 * b1).sum() \
                + reg * (W2 * W2).sum() \
                + reg * (b2 * b2).sum()
    prediction = Te.argmax(thpY, axis=1)

    #Define the Espilon value.
    esp = 10e-10

    # Update function for Cache and weights
    cacheW2_update = cacheW2 + Te.grad(cost_func, W2) * Te.grad(cost_func, W2)
    W2_update = W2 - learning_rate * Te.grad(cost_func, W2) / Te.sqrt(cacheW2 + esp)
    cacheb2_update = cacheb2 + Te.grad(cost_func, b2) * Te.grad(cost_func, b2)
    b2_update = b2 - learning_rate * Te.grad(cost_func, b2) / Te.sqrt(cacheb2 + esp)
    cacheW1_update = cacheW1 + Te.grad(cost_func, W1) * Te.grad(cost_func, W1)
    W1_update = W1 - learning_rate * Te.grad(cost_func, W1) / Te.sqrt(cacheW1 + esp)
    cacheb1_update = cacheb1 + Te.grad(cost_func, b1) * Te.grad(cost_func, b1)
    b1_update = b1 - learning_rate * Te.grad(cost_func, b1) / Te.sqrt(cacheb1 + esp)


    train = theano.function(
            inputs = [thX, thT],
            updates= [(W2, W2_update), (b2, b2_update),
                    (W1, W1_update), (b1, b1_update)] +
                    [(cacheW2, cacheW2_update), (cacheb2, cacheb2_update),
                    (cacheW1, cacheW1_update),(cacheb1, cacheb1_update)]
                    )

    train1 = theano.function(
            inputs = [thX, thT],
            updates=[(cacheW2, cacheW2_update), (cacheb2, cacheb2_update),
                    (cacheW1, cacheW1_update),(cacheb1, cacheb1_update)]
                    )

    train2 = theano.function(
            inputs = [thX, thT],
            updates= [(W2, W2_update), (b2, b2_update),
                    (W1, W1_update), (b1, b1_update)]
            )

    get_prediction = theano.function(
            inputs = [thX, thT],
            outputs= [cost_func, prediction]
    )

    costs_test = []
    check_value = []
    time = datetime.now()
    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size: (1 + batch) * batch_size]
            Ybatch = Ytrain[batch * batch_size: (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size: (1 + batch) * batch_size]

            train1(Xbatch, Tbatch)
            train2(Xbatch, Tbatch)

            if batch % 10 == 0:
                cost_val, prediction = get_prediction(Xtest, Ttest)
                acc = accuracy(Ytest, prediction)
                costs_test.append(cost_val)
                print("Epoch", epoch, "Cost", cost_val, "Acc", acc)

    print(datetime.now() - time)
    return costs_test

def train_rms(X, Y, T, Hidden_1, learning_rate, reg,
                epochs, batch_size, cache, decay):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    #We will create the 1 layer network.
    #Define all the parameters here.
    N, D = X.shape
    H1 = Hidden_1
    K = len(set(Y))
    batches = len(Ttrain) // batch_size

    #Initialize the weight variables
    W1_init = np.random.randn(D, H1) / np.sqrt(D)
    b1_init = np.zeros(H1)
    W2_init = np.random.randn(H1, K) / np.sqrt(H1)
    b2_init = np.zeros(K)

    #Setup the cache initial values.
    if cache == 1:
        cacheW1_init = np.ones((D, H1))
        cacheb1_init = np.ones(H1)
        cacheW2_init = np.ones((H1, K))
        cacheb2_init = np.ones(K)
    else:
        cacheW1_init = np.zeros((D, H1))
        cacheb1_init = np.zeros(H1)
        cacheW2_init = np.zeros((H1, K))
        cacheb2_init = np.zeros(K)

    #Setup the X & Y varilables along with shared theano variables.
    thX = Te.matrix("X")
    thT = Te.matrix("T")
    W1 = theano.shared(W1_init, "W1")
    b1 = theano.shared(b1_init, "b1")
    W2 = theano.shared(W2_init, "W2")
    b2 = theano.shared(b2_init, "b2")
    cacheW1 = theano.shared(cacheW1_init, "cacheW1")
    cacheb1 = theano.shared(cacheb1_init, "cacheb1")
    cacheW2 = theano.shared(cacheW2_init, "cacheW2")
    cacheb2 = theano.shared(cacheb2_init, "cacheb2")

    #Setup the neural network model.
    hidden = Te.nnet.relu(thX.dot(W1) + b1)
    thpY = Te.nnet.softmax(hidden.dot(W2) + b2)

    #Define the cost function and the prediction function
    cost_func = - (thT * Te.log(thpY)).sum() \
                + reg * ((W1 * W1).sum() \
                + (b1 * b1).sum() \
                + (W2 * W2).sum() \
                + (b2 * b2).sum()
                )
    prediction = Te.argmax(thpY, axis = 1)

    #Calculate the updates for all the weight and cache variables
    W1_grad = Te.grad(cost_func, W1)
    b1_grad = Te.grad(cost_func, b1)
    W2_grad = Te.grad(cost_func, W2)
    b2_grad = Te.grad(cost_func, b2)

    #Setup the espilon.
    esp = 10e-10

    cacheW1_update = decay * cacheW1 + (1 - decay) * W1_grad * W1_grad
    W1_udpate = W1 - learning_rate * W1_grad / Te.sqrt(cacheW1 + esp)
    cacheb1_update = decay * cacheb1 + (1 - decay) * b1_grad * b1_grad
    b1_update = b1 - learning_rate * b1_grad / Te.sqrt(cacheb1 + esp)
    cacheW2_update = decay * cacheW2 + (1 -decay) * W2_grad * W2_grad
    W2_update = W2 - learning_rate * W2_grad / Te.sqrt(cacheW2 + esp)
    cacheb2_update = decay * cacheb2 + (1 - decay) * b2_grad * b2_grad
    b2_update = b2 - learning_rate * b2_grad / Te.sqrt(cacheb2 + esp)

    train = theano.function(
            inputs=[thX, thT],
            updates=[(cacheW1, cacheW1_update), (cacheb1, cacheb1_update),
                    (cacheW2, cacheW2_update), (cacheb2, cacheb2_update),
                    (W1, W1_udpate), (b1, b1_update),
                    (W2, W2_update), (b2, b2_update)])
    #
    # train1 = theano.function(
    #         inputs=[thX, thT],
    #         updates=[(cacheW1, cacheW1_update), (cacheb1, cacheb1_update),
    #                 (cacheW2, cacheW2_update), (cacheb2, cacheb2_update)])
    #
    # train2 = theano.function(
    #         inputs=[thX, thT],
    #         updates=[(W1, W1_udpate), (b1, b1_update),
    #                 (W2, W2_update), (b2, b2_update)])


    get_prediction = theano.function(
            inputs=[thX, thT],
            outputs=[cost_func, prediction]
    )

    costs_test = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size : (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size : (1 + batch) * batch_size]

            train(Xbatch, Tbatch)
            #train2(Xbatch, Tbatch)

            if batch % 10 == 0:
                cost_val, pred = get_prediction(Xtest, Ttest)
                costs_test.append(cost_val)
                acc = accuracy(Ytest, pred)
                print("Epoch", epoch, "Cost", cost_val, "Acc", acc)

    return costs_test

def train_adam(X, Y, T, Hidden_1, learning_rate, reg, epochs,
                batch_size, beta1, beta2):


    Xtrain, Ytrain, Ttrain  = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    #We will create the 1 layer network.
    #Define all the parameters here.
    N, D = X.shape
    H1 = Hidden_1
    K = len(set(Y))
    batches = len(Ttrain) // batch_size

    #Initialize the weight variables
    W1_init = np.random.randn(D, H1) / np.sqrt(D)
    b1_init = np.zeros(H1)
    W2_init = np.random.randn(H1, K) / np.sqrt(H1)
    b2_init = np.zeros(K)

    #Define the momentum and the cache initial variables.
    mu_W1_init = np.zeros_like(W1_init)
    mu_b1_init = np.zeros_like(b1_init)
    mu_W2_init = np.zeros_like(W2_init)
    mu_b2_init = np.zeros_like(b2_init)
    cacheW1_init = np.zeros_like(W1_init)
    cacheb1_init = np.zeros_like(b1_init)
    cacheW2_init = np.zeros_like(W2_init)
    cacheb2_init = np.zeros_like(b2_init)

    ##Setup the X and Y placeholders and define the shared theano variables.
    thX = Te.matrix("X")
    thT = Te.matrix("T")
    W1 = theano.shared(W1_init, "W1")
    b1 = theano.shared(b1_init, "b1")
    W2 = theano.shared(W2_init, "W2")
    b2 = theano.shared(b2_init, "b2")

    mu_W1 = theano.shared(mu_W1_init, "W1")
    mu_b1 = theano.shared(mu_b1_init, "b1")
    mu_W2 = theano.shared(mu_W2_init, "W2")
    mu_b2 = theano.shared(mu_b2_init, "b2")

    cacheW1 = theano.shared(cacheW1_init, "W1")
    cacheb1 = theano.shared(cacheb1_init, "b1")
    cacheW2 = theano.shared(cacheW2_init, "W2")
    cacheb2 = theano.shared(cacheb2_init, "b2")

    #Define our neural network model.
    hidden = Te.nnet.relu(thX.dot(W1) + b1)
    thpY = Te.nnet.softmax(hidden.dot(W2) + b2)

    #Setup the cost function and the prediction function.
    cost_func = - (thT * Te.log(thpY)).sum() \
                + reg * ((W1 * W1).sum() \
                + (b1 * b1).sum() \
                + (W2 * W2).sum() \
                + (b2 * b2).sum()
                )
    prediction = Te.argmax(thpY, axis = 1)

    #Setup the gradiants.
    W1_grad = Te.grad(cost_func, W1)
    b1_grad = Te.grad(cost_func, b1)
    W2_grad = Te.grad(cost_func, W2)
    b2_grad = Te.grad(cost_func, b2)

    #Setup the Epsilon and the bias correction value
    esp = 10e-8
    t_init = 1
    t = theano.shared(t_init, "time")

    #Setup the update values.
    mu_W2_update = (beta1 * mu_W2 + (1 - beta1) * W2_grad)
    cacheW2_update = (beta2 * cacheW2 + (1 - beta2) * W2_grad * W2_grad)
    W2_update = W2 - learning_rate * Te.sqrt(1 - beta2 ** t) * mu_W2 / (Te.sqrt(cacheW2 + esp) * (1 - beta1 ** t))

    mu_b2_update = (beta1 * mu_b2 + (1 - beta1) * b2_grad)
    cacheb2_update = (beta2 * cacheb2 + (1 - beta2) * b2_grad * b2_grad)
    b2_update = b2 - learning_rate * Te.sqrt(1 - beta2 ** t) * mu_b2 / (Te.sqrt(cacheb2 + esp) * (1 - beta1 ** t))

    mu_W1_update = (beta1 * mu_W1 + (1 - beta1) * W1_grad)
    cacheW1_update = (beta2 * cacheW1 + (1 - beta2) * W1_grad * W1_grad)
    W1_update = W1 - learning_rate * Te.sqrt(1 - beta2 ** t) * mu_W1 / (Te.sqrt(cacheW1 + esp) * (1 - beta1 ** t))

    mu_b1_update = (beta1 * mu_b1 + (1 - beta1) * b1_grad)
    cacheb1_update = (beta2 * cacheb1 + (1 - beta2) * b1_grad * b1_grad)
    b1_update = b1 - learning_rate * Te.sqrt(1 - beta2 ** t) * mu_b1 / (Te.sqrt(cacheb1 + esp) * (1 - beta1 ** t))


    t_update = t + 1

    train = theano.function(
                    inputs = [thX, thT],
                    updates = [(W2, W2_update), (b2, b2_update),
                            (W1,W1_update), (b1, b1_update),
                            (mu_W2, mu_W2_update), (mu_b2, mu_b2_update),
                            (mu_W1, mu_W1_update), (mu_b1, mu_b1_update),
                            (cacheW2, cacheW2_update), (cacheb2, cacheb2_update),
                            (cacheW1, cacheW1_update), (cacheb1, cacheb1_update),
                            (t, t_update)]
                            )

    # train = theano.function(
    #                     inputs = [thX, thT],
    #                     updates = [(W1,W1_update), (b1, b1_update),
    #                               (W2, W2_update), (b2, b2_update)])

    get_prediction = theano.function(
                    inputs = [thX, thT],
                    outputs= [cost_func, prediction]
    )

    costs_test = []

    for epoch in range(epochs):
        for batch in range(batches):

            Xbatch = Xtrain[batch * batch_size : (1 + batch) * batch_size]
            Tbatch = Ttrain[batch * batch_size : (1 + batch) * batch_size]

            train(Xbatch, Tbatch)

            if batch % 10 == 0:
                cost_val, pred = get_prediction(Xtest, Ttest)
                costs_test.append(cost_val)
                acc = accuracy(Ytest, pred)
                print("Epoch", epoch, "Cost", cost_val, "Acc", acc)

    return costs_test

def main(Normal = True, Mom = True, Adagrad = True, rms = True,
        rms_m = True, Adam = True):
    X, Y, _, _ = U.get_data(PCA_data = True, num_parameters = 300)
    T = U.onehotencode(Y)

    if Normal:
        costs_normal = train_theano(X, Y, T,
                    Hidden_1 = 300, learning_rate = 4e-5,
                    reg = 0.01, epochs = 10, batch_size = 500)
        _ = plt.plot(costs_normal, label = "normal")

    if Mom:
        costs_momentum = train_momentum(X, Y, T,
                    Hidden_1 = 300, learning_rate = 4e-5,
                    reg = 0.01, epochs = 10, batch_size = 500,
                    mu_momentum = 0.95)
        _ = plt.plot(costs_momentum, label = "Momentum = 0.9")

    if Adagrad:
        costs_adagrad = train_Adagrad(X, Y, T,
                    Hidden_1 = 300, learning_rate = 4e-4,
                    reg = 0.01, epochs = 10, batch_size = 500, cache = 1)
        _ = plt.plot(costs_adagrad, label = "Adagrad")

    if rms:
        costs_rms = train_rms(X, Y, T,
                    Hidden_1 = 300, learning_rate = 8e-4,
                    reg = 0.01, epochs = 10, batch_size = 500, cache = 1,
                    decay = 0.99)
        _ = plt.plot(costs_rms, label = "RMS")

    if Adam:
        costs_adam = train_adam(X, Y, T,
                    Hidden_1 = 300, learning_rate = 10e-4,
                    reg = 0.01, epochs = 10, batch_size = 500,
                    beta1 = 0.9, beta2 = 0.999)
        _ = plt.plot(costs_adam, label = "ADAM")


    _ = plt.legend()
    plt.show()

if __name__ == "__main__":
    main(Normal = True, Mom = True, Adagrad = True, rms = True,
            rms_m = False, Adam = True)
