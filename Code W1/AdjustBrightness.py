import cv2

file_path = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/Lena_256bmp.bmp'
alpha = 1

image = cv2.imread(file_path)
imageYCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

imageYCC[:,:,0] = 255*(imageYCC[:,:,0]/255)**alpha

image_output = cv2.cvtColor(imageYCC, cv2.COLOR_YCR_CB2BGR)

cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/Result/LenaDark.bmp', image_output)
cv2.imshow("test", image_output/255)
cv2.waitKey(0)
cv2.destroyAllWindows()