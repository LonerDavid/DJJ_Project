import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def gaussian_kernel(size):
    m, n = np.meshgrid(range(-size, size+1), range(-size, size+1))
    kernel = np.exp(-0.1 * (m**2 + n**2))
    return kernel / np.sum(kernel)

def blur_image(image, kernel, noise_level):
    blurred = convolve2d(image, kernel, mode='same', boundary='wrap')
    noise = np.random.normal(0, noise_level, image.shape)
    return blurred + noise
    # return blurred

def equalize_image(blurred_image, kernel, c):
    F_blurred = np.fft.fft2(blurred_image)
    F_kernel = np.fft.fft2(kernel, s=blurred_image.shape)
    
    # H = np.conj(F_kernel) / (np.abs(F_kernel)**2 + c)
    H = 1 / (( c / np.conj(F_kernel)) + F_kernel)
    
    F_reconstructed = F_blurred * H
    reconstructed_image = np.fft.ifft2(F_reconstructed).real
    return reconstructed_image

# image = np.tile(np.linspace(0, 255, 256), (256, 1))
file_path = "/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/Peppers.png"
image = plt.imread(file_path)

kernel = gaussian_kernel(size=10)
blurred_image = blur_image(image, kernel, noise_level=0.1)

c1 = 0.01
c2 = 0.1

reconstructed_c1 = equalize_image(blurred_image, kernel, c=c1)
reconstructed_c2 = equalize_image(blurred_image, kernel, c=c2)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(2, 2, 2)
plt.title("Blurred Image")
plt.imshow(blurred_image, cmap='gray')
plt.subplot(2, 2, 3)
plt.title("Reconstructed Image (C=" + str(c1) + ")")
plt.imshow(reconstructed_c1, cmap='gray')
plt.subplot(2, 2, 4)
plt.title("Reconstructed Image (C=" + str(c2) + ")")
plt.imshow(reconstructed_c2, cmap='gray')
plt.show()
