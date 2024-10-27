import numpy as np
import matplotlib.pyplot as plt
import math

n = np.arange(-50, 101)
an = 1.5
L = 5
sigma = 1

x1 = 0.1 * n + an * (np.random.rand(len(n)) - 0.5)

def h (sigma, L):
    h = np.empty(2*L+1, dtype=float)
    C_sum = 0
    for i in range(1, L+1):
        C_sum += math.exp(-1 * sigma * abs(i))

    C = 1 / C_sum
    for i in range(2*L+1):
        h[i] = C * math.exp(-1 * sigma * abs(i-L))

    return h

smooth = np.convolve(x1, h(sigma, L), mode='same')
fig, axs = plt.subplots(2, 1, figsize=(16,9))

axs[0].stem(n, x1)
axs[0].set_title('Orignal Signal')
axs[1].stem(n, smooth)
axs[1].set_title('Smoother Signal')

plt.suptitle("Smoother (an=" + str(an) + ", σ=" + str(sigma) + ")")
plt.show()