import numpy as np
# import matplotlib.pyplot as plt


def plot_hist(data, plot_zeros=True):
    plt.figure()
    c, b = np.histogram(data, bins=10)
    if not plot_zeros:
        for i in range(len(b)):
            if b[i] >= 0:
                c[i-1] = 0
                break
        print(c, b)
    plt.hist(b[:-1], b, weights=c)
    plt.draw()

