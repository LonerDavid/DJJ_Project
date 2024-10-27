import numpy as np
import matplotlib.pyplot as plt
import math

n = np.arange(-30, 100)
an = 0.2
L = 10
sigma = 0.2

x = np.zeros_like(n, dtype=float)
x[(n >= -10) & (n <= 20)] = 1
x[(n >= 50) & (n <= 80)] = 1

noise = an * (np.random.rand(len(n)) - 0.5)
x1 = x + noise
# x1 = x

def h (sigma, L):
    h = np.empty(2*L+1, dtype=float)
    C_sum = 0
    for i in range(1, L+1):
        C_sum += math.exp(-1 * sigma * abs(i))

    C = 1 / C_sum
    for i in range(2*L+1):
        if i < L:
            h[i] = -1 * C * math.exp(-1 * sigma * abs(i-L))
        elif i > L:
            h[i] = C * math.exp(-1 * sigma * abs(i-L))
        else:
            h[i] = 0
    
    return h

edges = np.convolve(x1, h(sigma, L), mode='same')
fig, axs = plt.subplots(2, 1, figsize=(16,9))

axs[0].stem(n, x1)
axs[0].set_title('Orignal Signal')
axs[1].stem(n, edges)
axs[1].set_title('Edge Detection Signal')

plt.suptitle("Edge Detection Filter (an=" + str(an) + ", σ=" + str(sigma) + ")")
plt.show()