import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path_1 = "/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/TIFFANY.BMP"
file_path_2 = "/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/Lena256c.bmp"

image1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(file_path_2, cv2.IMREAD_GRAYSCALE)

# Width of the mask
th = 20 

# FFT
f1 = np.fft.fft2(image1)
f2 = np.fft.fft2(image2)

# Shift the zero frequency component to the center
f1_shift = np.fft.fftshift(f1)
f2_shift = np.fft.fftshift(f2)

rows, cols = image1.shape
c_row, c_col = rows // 2, cols // 2

# Low-pass mask
low_pass = np.zeros((rows, cols), np.uint8)
low_pass[c_row-th:c_row+th, c_col-th:c_col+th] = 1

# High-pass mask
high_pass = 1 - low_pass

low_freq = f1_shift * low_pass
high_freq = f2_shift * high_pass
combined = low_freq+ high_freq

# IFFT to get the final image
f_ishift = np.fft.ifftshift(combined)
fused_image = np.fft.ifft2(f_ishift)
fused_image = np.abs(fused_image)

plt.subplot(131), plt.imshow(image1, cmap='gray'), plt.title('Image 1(Low Pass)')
plt.subplot(132), plt.imshow(image2, cmap='gray'), plt.title('Image 2(High Pass)')
plt.subplot(133), plt.imshow(fused_image, cmap='gray'), plt.title('Fused Image')
plt.show()
