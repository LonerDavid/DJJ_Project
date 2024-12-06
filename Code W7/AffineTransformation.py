import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def rotate(input_image, center, deg):
    M1, N1 = input_image.shape
    M2, N2 = M1, N1 #h, w
    new_image = np.zeros((M2, N2))
    cm = center[0]
    cn = center[1]

    rad = deg / 180 * math.pi
    cos = math.cos(rad)
    sin = math.sin(rad)

    for m2 in range(M2):
        for n2 in range(N2):
            m1 = cos*(m2-cm) - sin*(n2-cn) + cm
            n1 = sin*(m2-cm) + cos*(n2-cn) + cn

            if m1<=0 or m1>=M1-1 or n1<=0 or n1>=N1-1:
                new_image[m2, n2] = 0
            else:
                m0, n0 = int(np.floor(m1)), int(np.floor(n1))
                a = m1 - m0
                b = n1 - n0

                new_image[m2, n2] = ((1-a)*(1-b)*input_image[m0, n0] + 
                                     a*(1-b)*input_image[m0+1, n0] +
                                     (1-a)*b*input_image[m0, n0+1] +
                                     a*b*input_image[m0+1, n0+1])
    
    return new_image

def shearing(input_image, center, axis, eta):
    M1, N1 = input_image.shape
    M2, N2 = M1, N1 #h, w
    new_image = np.zeros((M2, N2))
    cm = center[0]
    cn = center[1]
 
    if axis == 'y':
        for m2 in range(M2):
            for n2 in range(N2):
                m1 = m2- eta*(n2-cn)
                n1 = n2

                if m1<0 or m1>=M1-1 or n1<0 or n1>=N1-1:
                    new_image[m2, n2] = 0
                else:
                    m0, n0 = int(np.floor(m1)), int(np.floor(n1))
                    a = m1 - m0
                    b = n1 - n0

                    new_image[m2, n2] = ((1-a)*(1-b)*input_image[m0, n0] + 
                                        a*(1-b)*input_image[m0+1, n0] +
                                        (1-a)*b*input_image[m0, n0+1] +
                                        a*b*input_image[m0+1, n0+1])
    else: #default x-axis
        for m2 in range(M2):
            for n2 in range(N2):
                m1 = m2
                n1 = -eta*(m2-cm) + n2

                if m1<0 or m1>=M1-1 or n1<0 or n1>=N1-1:
                    new_image[m2, n2] = 0
                else:
                    m0, n0 = int(np.floor(m1)), int(np.floor(n1))
                    a = m1 - m0
                    b = n1 - n0

                    new_image[m2, n2] = ((1-a)*(1-b)*input_image[m0, n0] + 
                                        a*(1-b)*input_image[m0+1, n0] +
                                        (1-a)*b*input_image[m0, n0+1] +
                                        a*b*input_image[m0+1, n0+1])

    return new_image

file_path = "/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/PEPPER.BMP"
image = cv2.imread(file_path)
test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
center = (image.shape[0]*0.5-0.5, image.shape[1]*0.5-0.5)
rotated_image = rotate(test_image, center, 30)
sheared_image = shearing(test_image, center, "x", 0.3)

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(test_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Rotated Image")
plt.imshow(rotated_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Sheared Image")
plt.imshow(sheared_image, cmap='gray')

plt.show()