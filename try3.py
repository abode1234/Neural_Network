import numpy as np
import matplotlib.pyplot as plt

# A
a = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]

# B
b = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0]

# C
c = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]

y = [[1, 0, 0],   # A
     [0, 1, 0],   # B
     [0, 0, 1]]   # C


plt.imshow(np.array(a).reshape(5, 6))
plt.title("Letter A")
plt.show()

plt.imshow(np.array(b).reshape(5, 6))
plt.title("Letter B")
plt.show()

plt.imshow(np.array(c).reshape(5, 6))
plt.title("Letter C")
plt.show()

x = [np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), np.array(c).reshape(1, 30)]
x = np.array(x)

y = np.array(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_forward(x, w1, w2):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)

    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    return a2

def generate_wt(x, y):
    return np.random.randn(x, y)

def loss(out, Y):
    s = np.square(out - Y)
    return np.sum(s) / len(Y)

def back_prop(x, y, w1, w2, alpha):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)

    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    d2 = a2 - y
    d1 = np.multiply((w2.dot(d2.transpose())).transpose(), np.multiply(a1, 1 - a1))

    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)

    w1 -= alpha * w1_adj
    w2 -= alpha * w2_adj

    return w1, w2

def train(x, Y, w1, w2, alpha=0.01, epoch=10):
    acc = []
    losss = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append(loss(out, Y[i]))
            w1, w2 = back_prop(x[i], Y[i], w1, w2, alpha)
        print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(x))) * 100)
        acc.append((1 - (sum(l) / len(x))) * 100)
        losss.append(sum(l) / len(x))
    return acc, losss, w1, w2

def predict(x, w1, w2):
    out = f_forward(x, w1, w2)
    maxm = 0
    k = 0
    for i in range(len(out[0])):
        if maxm < out[0][i]:
            maxm = out[0][i]
            k = i
    if k == 0:
        print("Image is of letter A.")
    elif k == 1:
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")
    plt.imshow(x.reshape(5, 6))
    plt.show()

w1 = generate_wt(30, 5)
w2 = generate_wt(5, 3)

acc, losss, w1, w2 = train(x, y, w1, w2, alpha=0.01, epoch=100)

predict(np.array(a).reshape(1, 30), w1, w2)
