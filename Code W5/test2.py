import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/peppers.bmp'  # Replace with your image path
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float32
gray = np.float32(gray)

# Compute gradients
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

# Compute products of derivatives
Ixx = Ix**2
Iyy = Iy**2
Ixy = Ix * Iy

# Apply Gaussian filter to smooth the products
Ixx = cv2.GaussianBlur(Ixx, (5, 5), sigmaX=1)
Iyy = cv2.GaussianBlur(Iyy, (5, 5), sigmaX=1)
Ixy = cv2.GaussianBlur(Ixy, (5, 5), sigmaX=1)

# Compute Harris response matrix
k = 0.04  # Harris detector free parameter
detM = (Ixx * Iyy) - (Ixy**2)  # Determinant of the matrix
traceM = Ixx + Iyy  # Trace of the matrix
R = detM - k * (traceM**2)  # Harris response

# Threshold to identify corners
threshold = 0.01 * R.max()
corners = np.zeros_like(gray)
corners[R > threshold] = 255

# Mark the corners on the original image
img_corners = img.copy()
img_corners[R > threshold] = [0, 0, 255]  # Red corners

# Display the result
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
plt.title("Corners Detected")
plt.axis("off")
plt.show()

