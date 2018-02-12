import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils_2 as U

def accuracy(Y, pY):
    return np.mean(Y == pY)

def train_normal(X, Y, T, Hidden_1, learning_rate, epochs, batch_size, optim):

    Xtrain, Ytrain, Ttrain = X[:-1000], Y[:-1000], T[:-1000]
    Xtest, Ytest, Ttest = X[-1000:], Y[-1000:], T[-1000:]

    #Get the shape paramaters.
    N, D = X.shape
    H1 = Hidden_1
    K = len(set(Y))
    batches = len(Ytrain)//batch_size

    #Initialize the weights.
    W1_init = np.random.randn(D, H1) / np.sqrt(D)
    b1_init = np.zeros(H1)
    W2_init = np.random.randn(H1, K) / np.sqrt(H1)
    b2_init = np.zeros(K)


    #Setup the placeholders.
    X = tf.placeholder(tf.float32, shape=(None, D), name="X")
    T = tf.placeholder(tf.float32, shape=(None, K), name="T")
    W1 = tf.Variable(initial_value=W1_init.astype(np.float32), trainable=True, name="W1")
    b1 = tf.Variable(initial_value=b1_init.astype(np.float32), trainable=True, name="b1")
    W2 = tf.Variable(initial_value=W2_init.astype(np.float32), trainable=True, name="W2")
    b2 = tf.Variable(initial_value=b2_init.astype(np.float32), trainable=True, name="b2")

    #Develop the neural network.
    hidden = tf.nn.relu(tf.matmul(X, W1) + b1)
    Tish = tf.matmul(hidden, W2) + b2

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Tish, labels=T))

    if optim == "Normal":
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    elif optim == "Momentum":
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost)
    elif optim == "Adagrad":
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
    elif optim == "rms":
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, epsilon=1e-8).minimize(cost)
    elif optim == "rms_m":
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.95, epsilon=1e-8).minimize(cost)
    elif optim == "Adam":
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost)



    prediction = tf.argmax(Tish, axis=1)

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            for batch in range(batches):

                Xbatch = Xtrain[batch * batch_size:(1 + batch) * batch_size,]
                Tbatch = Ttrain[batch * batch_size:(1 + batch) * batch_size,]

                sess.run(train_op, feed_dict={X: Xbatch, T: Tbatch})

                if batch % 10 == 0:
                    cost_val = sess.run(cost, feed_dict={X: Xtest, T:Ttest})
                    pred = sess.run(prediction, feed_dict={X:Xtest, T:Ttest})
                    costs.append(cost_val)
                    acc = accuracy(Ytest, pred)
                    print("Epoch", epoch, "Cost", cost_val, "Acc", acc)

    return costs

def main(Normal = True, Mom = True, Adagrad = True, rms = True,
        rms_m = False, Adam = True):

    X, Y, _, _ = U.get_data(PCA_data = False, num_parameters = 200)
    T = U.onehotencode(Y)

    if Normal:
    costs_normal = train_normal(X, Y, T, Hidden_1=300,
                    learning_rate = 10e-4, epochs=10, batch_size=500, optim="Normal")
    _ = plt.plot(costs_normal, label="NormalGD")

    if Mom:
    costs_mom = train_normal(X, Y, T, Hidden_1=300,
                    learning_rate = 10e-5, epochs=10, batch_size=500, optim="Momentum")
    _ = plt.plot(costs_mom, label="Momentum")

    if Adagrad:
    costs_Adagrad = train_normal(X, Y, T, Hidden_1=300,
                    learning_rate = 10e-3, epochs=10, batch_size=500, optim="Adagrad")
    _ = plt.plot(costs_Adagrad, label="Adagrad")

    if rms:
    costs_rms = train_normal(X, Y, T, Hidden_1=300,
                    learning_rate = 10e-4, epochs=10, batch_size=500, optim="rms")
    _ = plt.plot(costs_rms, label="RMS")

    if rms_m:
    costs_rms_m = train_normal(X, Y, T, Hidden_1=300,
                    learning_rate = 10e-5, epochs=10, batch_size=500, optim="rms_m")
    _ = plt.plot(costs_rms_m, label="RMS-M")

    if Adam:
    costs_Adam = train_normal(X, Y, T, Hidden_1=300,
                    learning_rate = 10e-4, epochs=10, batch_size=500, optim="Adam")
    _ = plt.plot(costs_Adam, label="Adam")


    _ = plt.legend()
    plt.show()

if __name__ == "__main__":
    main(Normal = True, Mom = True, Adagrad = True, rms = True,
            rms_m = False, Adam = True)
