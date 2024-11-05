import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-10, 10, 100)
s = np.linspace(-10, 10, 100)
data = np.array([(2, -1, 3), (-1, 3, 5), (0, 2, 4), (4, -2, -1), (1, 0, 4), (-2, 5, 5)
])
mean_vec = np.mean(data, axis=0)
centered_data = data - mean_vec

U,S,Vh = np.linalg.svd(centered_data)

#1-dimension
x = mean_vec[0] + t * Vh[0, 0]
y = mean_vec[1] + t * Vh[0, 1]
z = mean_vec[2] + t * Vh[0, 2]

#2-dimension
T, S = np.meshgrid(t, s)
X = mean_vec[0] + T * Vh[0, 0] + S * Vh[1, 0]
Y = mean_vec[1] + T * Vh[0, 1] + S * Vh[1, 1]
Z = mean_vec[2] + T * Vh[0, 2] + S * Vh[1, 2]

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z, label="parametric curve")
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', label='parametric surface')
ax.scatter(data[:,0], data[:,1], data[:,2])
ax.set_xlabel('x',fontdict={'size':15},labelpad=8, color='#f00')  # 設定 x 軸標題
ax.set_ylabel('y',fontdict={'size':15},labelpad=8, color='#0f0')  # 設定 y 軸標題
ax.set_zlabel('z',fontdict={'size':15},labelpad=8, color='#00f')  # 設定 z 軸標題
ax.legend()

plt.show()