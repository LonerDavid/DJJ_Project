import numpy as np
import matplotlib.pyplot as plt
import math

def bilateral_filter(signal, k1, k2, L):
    filtered_signal = np.zeros_like(signal, dtype=np.float32)
    length = len(signal)
    
    for i in range(length):
        numerator = 0
        C = 0

        if i-L < 0:
            for j in range(0, i+L+1):
                weight = np.exp(-((i - j)**2) * k1) * np.exp(-((signal[i]-signal[j])**2) * k2)
                numerator += weight * signal[j]
                C += weight
        elif i+L >= length:
            for j in range(i-L, length):
                weight = np.exp(-((i - j)**2) * k1) * np.exp(-((signal[i]-signal[j])**2) * k2)
                numerator += weight * signal[j]
                C += weight
        else:
            for j in range(i-L, i+L+1):
                weight = np.exp(-((i - j)**2) * k1) * np.exp(-((signal[i]-signal[j])**2) * k2)
                numerator += weight * signal[j]
                C += weight

        filtered_signal[i] = numerator / C if C != 0 else 0
    
    return filtered_signal

n = np.arange(0, 100)
an = 0.3
L = 20
k1 = 0.1
k2 = 5

x = np.zeros_like(n, dtype=float)
x[(n >= 0) & (n <= 50)] = 1
x[(n > 50) & (n <= 100)] = 0

noise = an * (np.random.rand(len(n)) - 0.5)
x1 = x + noise

x0 = bilateral_filter(x1, k1, k2, L)

plt.figure(figsize=(12, 6))
plt.plot(x1, label='Original Signal', color='orange')
plt.plot(x0, label='Filtered Signal', color='green')
plt.legend()
plt.title("Bilateral Filtering")
# plt.xlabel("Index")
# plt.ylabel("Amplitude")
plt.grid(True)
plt.show()