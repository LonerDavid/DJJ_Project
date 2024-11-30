import numpy as np
import matplotlib.pyplot as plt

size = (100, 100)
x0, y0 = 50, 50
rx, ry = 30, 20

x, y = np.meshgrid(range(size[0]), range(size[1]))
ellipse = ((x - x0)**2 / rx**2 + (y - y0)**2 / ry**2) <= 1
image = ellipse.astype(int)

#Norm
L0 = np.count_nonzero(image)
L1 = np.sum(image)
L2 = np.sqrt(np.sum(image**2))
Linf = np.max(image)

#Raw moment
m_00 = np.sum(image)
m_10 = np.sum(x * image)
m_01 = np.sum(y * image)
x_bar = m_10 / m_00
y_bar = m_01 / m_00

#Central Moment
mu_20 = np.sum((x - x_bar)**2 * image) / m_00
mu_11 = np.sum((x - x_bar) * (y - y_bar) * image) / m_00
mu_02 = np.sum((y - y_bar)**2 * image) / m_00

print("L0 = " , L0)
print("L1 = " , L1)
print("L2 = " , L2)
print("Linf = " , Linf)

print("mu_20 = " , mu_20)
print("mu_11 = " , mu_11)
print("mu_02 = " , mu_02)


plt.figure(figsize=(6,6))
plt.imshow(image, cmap="gray")
plt.title("Elliptic Image")
plt.show()