import numpy as np
import matplotlib.pyplot as plt

# Define the length of each segment
L = 20

# 0. Spacing Signal
zero = np.zeros(int(L))

# 1. Pulse Signal
pulse = np.ones(L)

# 2. Linearly Increasing Signal with slope 1/10
line_pos = np.linspace(-1, 1, L, True)

# 3. Linearly Decreasing Signal with slope -1/10
line_neg = np.linspace(1, -1, L, True)

# 4. Sine Wave with period 20
x = np.arange(L)
sine_wave = np.sin(2 * np.pi * x / 20)

# Combine all segments into one signal
x = np.concatenate([zero, pulse, zero, line_pos, zero, line_neg, zero, sine_wave, zero])

h = np.linspace(-1, 1, L, True)

result = np.zeros(len(x))
length = len(h)
for i in range(0, len(x) - len(h)):
    arr1 = np.array(x)[i:i+length]
    result[i] = np.corrcoef(arr1, h)[1, 0]

h_display = np.concatenate([h, np.zeros(int(len(x)-len(h)))])

fig, axs = plt.subplots(3, 1, figsize=(24,9))

axs[0].stem(x)
axs[0].set_title('Orignal Signal')
axs[1].stem(h_display)
axs[1].set_title('Desired Pattern')
axs[2].stem(result)
axs[2].set_ylim([0.8, 1])
axs[2].set_title('Matched Filter Signal')

plt.suptitle("Matched Filter")
plt.show()