import numpy as np
import matplotlib.pyplot as plt
import cv2

def Interpolation(image, m_scale, n_scale):
    #1: original image, 2: new image
    M1, N1 = image.shape
    M2, N2 = int(M1 * m_scale), int(N1 * n_scale)
    new_image = np.zeros((M2, N2))

    for m2 in range(M2):
        for n2 in range(N2):
            m1 = m2 / m_scale
            n1 = n2 / n_scale

            #Boundary
            if m1<0 or m1>M1-1 or n1<0 or n1>N1-1:
                new_image[m2, n2] = 0
            
            else:
                m0, n0 = int(np.floor(m1)), int(np.floor(n1))
                a = m1 - m0
                b = n1 - n0

                new_image[m2, n2] = ((1-a)*(1-b)*image[m0, n0] + 
                                     a*(1-b)*image[m0+1, n0] +
                                     (1-a)*b*image[m0, n0+1] +
                                     a*b*image[m0+1, n0+1])
    
    return new_image


file_path = "/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/peppers.bmp"
image = cv2.imread(file_path)
test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
m, n = 1.5, 1.6

resized_image = Interpolation(test_image, m, n)

# Visualize original and resized images
print(test_image.shape)
print(resized_image.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(test_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Resized Image")
plt.imshow(resized_image, cmap='gray')

plt.show()
