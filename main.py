# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.sparse import random
import scipy.sparse as sparse
from scipy.sparse import coo_matrix


def calculateSumXi(x, H, i):
    sum = 0
    for j in range(0, len(x)):
        if j != i:
            sum += x[j] * H[j][i]
    return sum
#

def coordinate_descent(H, g, a, b, n):
    x = np.zeros(len(g))
    for l in range(0, n):
        for i in range(0, len(g)):
            x[i] = - ((calculateSumXi(x, H, i) - g[i]) / H[i][i])
            x[i] = np.clip(x[i], a[i], b[i])
        minimizeCheck = 0.5 * x.T @ H @ x - x.T @ g
        if abs(minimizeCheck) < 0.001:
            break
    return x


def steepest_descent(f, n, m):
    x = np.zeros((2, 1))
    result_x = []
    #for mu in mues:
    for i in range(0, n):
        (minimize, fgrad) = penaltyMethod(x, f, m)
        result_x.append(minimize)
        d = -np.asarray(fgrad)
        a = Armijo(x, minimize, fgrad, m, d, n, 1, 0.5, 0.0001)
        #TODO: check the stoping condition
        x = x + a*d
    return x, result_x


def Armijo(x, Fx, Fgrad, m, d, n, a, b, c):
    x = np.reshape(x, (len(x), 1))
    for i in range (0, n):
        xad = x[:, 0] + (a * d)[:, 0]
        (minimize_xad, fad_grad) = penaltyMethod(xad, f, m)
        check = np.asarray(Fx) + c * a * np.transpose(Fgrad) @ np.asarray(d)
        cond = minimize_xad <= check
        if (cond):
            return a
        else:
            a = b * a
    return a


def f(x1, x2):
    return (x1+x2)**2 - 10 * (x1 + x2)


def cieq(x1, x2):
    cieq1 = (x1**2 + x2**2 - 5)
    cieq2 = -x1
    return [cieq1, cieq2]


def ceq(x1, x2):
    return (3 * x1 + x2 - 6)**2


def gradient(x1, x2, m):
    # calculate gradient grad[x1, x2]
    grad = [0, 0]
    grad[0] = 2 * (x1 + x2) - 10 + 6 * m * (3 * x1 + x2 - 6)
    grad[1] = 2 * (x1 + x2) - 10 + 2 * m * (3 * x1 + x2 - 6)
    if max(0, cieq(x1, x2)[0]) != 0:
        grad[0] += 4 * m * x1 * (cieq(x1, x2)[0])
        grad[1] += 4 * m * x2 * (cieq(x1, x2)[0])

    if max(0, cieq(x1, x2)[1]) != 0:
        grad[0] += m * 2 * x1  # check the minus before
        grad[1] += 0  # erase
    return grad


def penaltyMethod(x, f, m):
    x1 = x[0]
    x2 = x[1]
    minimize = f(x1, x2)
    minimize += m * (ceq(x1, x2) + max(0, cieq(x1, x2)[0])**2 + max(0, cieq(x1, x2)[1])**2)
    grad = gradient(x1, x2, m)
    return minimize, grad


def f_4b(w, A, b, lamda, C):
    ones = np.ones(len(w)).T
    return np.linalg.norm(A @ C @ w - b)**2 + lamda * ones @ w


def grad_4b(w, A, b, lamda, C):
    w = np.reshape(w, (len(w), 1))
    return 2*(C.T @ A.T) @ (A @ C @ w - b) + lamda


def steepest_descent4b(A, b, C, lamda, n):
    w = np.zeros((400,1))
    w = np.reshape(w, (len(w), 1))
    result_w = []
    for i in range(0, n):
        Fw = f_4b(w, A, b, lamda, C)
        fgrad = grad_4b(w, A, b, lamda, C)
        fgrad = fgrad / np.linalg.norm(fgrad)
        result_w.append(Fw)
        d = -np.asarray(fgrad)
        a = Armijo4b(w, Fw, fgrad, d, 5, A, b, lamda, C)
        w += a * d
        w = np.clip(w, 0, None)
        w = np.reshape(w, (len(w), 1))
        #TODO: check the stoping condition
    return w, result_w


def Armijo4b(w, Fw, Fgrad, d, n, A, b, lamda, C):
    a = 1
    for i in range (0, n):
        wad = w[:, 0] + (a * d)[:, 0]
        wad = np.clip(wad, 0, None)
        minimizeWad = f_4b(wad, A, b, lamda, C)
        check = Fw + 0.0001 * a * np.transpose(Fgrad) @ np.asarray(d)
        cond = minimizeWad <= check
        if (cond):
            return a
        else:
            a = 0.5 * a
    return a

if __name__ == '__main__':
    #section 3c+d
    H = np.asarray([[5, -1, -1, -1, -1], [-1, 5, -1, -1, -1],
                    [-1, -1, 5, -1, -1], [-1, -1, -1, 5, -1],
                    [-1, -1, -1, -1, 5]])
    g = np.asarray([18, 6, -12, -6, 18])
    a = np.asarray([0, 0, 0, 0, 0])
    b = np.asarray([5, 5, 5, 5, 5])
    x = coordinate_descent(H, g, a, b, 20)
    print(x)

    #section 2d
    mues = [0.01, 0.1, 1, 10, 100]
    for m in mues:
        x, resultX = steepest_descent(f, 90, m)
        plt.plot(np.arange(0, len(resultX)), resultX)
        plt.title("minimized F(x1, x2) value for m =" + str(m) + "\n" +
                  "x* = " + str(x.T))
        plt.show()
    print(x)
    print(resultX)

    # #section 4b
    A = np.random.normal(size=(100, 200))
    x = sparse.random(200, 1, 0.1)
    b = A @ x + np.random.normal(scale=0.1, size=(100, 1))
    C = np.concatenate([np.eye(200), -np.eye(200)], axis=1)
    lamda = [10, 20, 30, 40, 50, 60, 70, 80 ,90, 100]
    # result_lamda = []
    # for l in lamda:
    #     wk, result_wk = steepest_descent4b(A, b, C, l, 50)
    #     xk = C @ wk
    #     numOfZeroesP = np.count_nonzero(xk)/(int)(len(xk))
    #     print (numOfZeroesP)
    #     result_lamda.append(numOfZeroesP)
    # plt.plot(lamda, result_lamda)
    # plt.title("number of non-zeros percentage")
    # plt.xlabel("lamda")
    # plt.ylabel("non-zeros percentage")
    # plt.show()


    #
    wk, result_wk = steepest_descent4b(A, b, C, 90, 100)
    xk = C @ wk
    numOfZeroesP = np.count_nonzero(xk)
    plt.plot(np.arange(0, len(result_wk)), result_wk)
    plt.title("Objective values")
    plt.legend(["Fw"])
    plt.xlabel("iterations")
    plt.ylabel("objective")
    print(np.linalg.norm(xk - x))
    plt.plot(np.arange(0, len(result_wk)), result_wk)
    plt.show()

    #


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
