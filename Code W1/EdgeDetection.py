import numpy as np
# import matplotlib.pyplot as plt
import cv2

file_path = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/peppers.bmp'

matrix1 = np.array([(1, 0, -1),(2, 0, -2),(1, 0, -1)], dtype='f') #Sobel horizontal
matrix2 = np.array([(1, 2, 1),(0, 0, 0),(-1, -2, -1)], dtype='f') #Sobel vertical
matrix3 = np.array([(0, -1, -2),(1, 0, -1),(2, 1, 0)], dtype='f') #Sobel 45 degree
matrix4 = np.array([(-2, -1, 0),(-1, 0, 1),(0, 1, 2)], dtype='f') #Sobel 135 degree
matrix5 = np.array([(-1, -1, -1),(-1, 8, -1),(-1, -1, -1)], dtype='f') #Laplacian Operator

c = 5  #constant for Simple
cs = 5 #constant for Sobel
cl = 8  #constant for Laplacian

image = cv2.imread(file_path)

print("Image Shape:", image.shape)


def Simple(index, image):
    rows, cols,_ = image.shape
    image1 = image * 0
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if index == 1:
        for i in range(0, rows):
            for j in range(1, cols):
                image1[i][j] = c * abs(grayImg[i][j]*1.0 - grayImg[i][j-1]*1.0)        
        cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Simple_hor.bmp', image1)
        return image1
    
    elif index == 2:
        for i in range(1, rows):
            for j in range(0, cols):
                image1[i][j] = c * abs(grayImg[i][j]*1.0 - grayImg[i-1][j]*1.0)
        cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Simple_ver.bmp', image1)
        return image1
    
    else:
        return image1
    

def Sobel(index, image):
    rows, cols,_ = image.shape
    image1 = image * 0
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if index == 1:
        for i in range(1,rows-1):
            for j in range(1, cols-1):
                a = np.array([(grayImg[i-1,j-1],grayImg[i-1,j],grayImg[i-1,j+1]),
                        (grayImg[i,j-1],  grayImg[i,j],  grayImg[i,j+1]),
                        (grayImg[i+1,j-1],grayImg[i+1,j],grayImg[i+1,j+1])])
                image1[i][j] = cs * abs(np.sum(np.multiply(a, matrix1)) * 0.25)
        cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Sobel1.bmp',image1)
        return image1
    
    elif index == 2:
        for i in range(1,rows-1):
            for j in range(1, cols-1):
                a = np.array([(grayImg[i-1,j-1],grayImg[i-1,j],grayImg[i-1,j+1]),
                        (grayImg[i,j-1],  grayImg[i,j],  grayImg[i,j+1]),
                        (grayImg[i+1,j-1],grayImg[i+1,j],grayImg[i+1,j+1])])
                image1[i][j] = cs * abs(np.sum(np.multiply(a, matrix2)) * 0.25)
        cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Sobel2.bmp',image1)
        return image1

    elif index == 3:
        for i in range(1,rows-1):
            for j in range(1, cols-1):
                a = np.array([(grayImg[i-1,j-1],grayImg[i-1,j],grayImg[i-1,j+1]),
                        (grayImg[i,j-1],  grayImg[i,j],  grayImg[i,j+1]),
                        (grayImg[i+1,j-1],grayImg[i+1,j],grayImg[i+1,j+1])])
                image1[i][j] = cs * abs(np.sum(np.multiply(a, matrix3)) * 0.25)
        cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Sobel3.bmp',image1)
        return image1
    
    elif index == 4:
        for i in range(1,rows-1):
            for j in range(1, cols-1):
                a = np.array([(grayImg[i-1,j-1],grayImg[i-1,j],grayImg[i-1,j+1]),
                        (grayImg[i,j-1],  grayImg[i,j],  grayImg[i,j+1]),
                        (grayImg[i+1,j-1],grayImg[i+1,j],grayImg[i+1,j+1])])
                image1[i][j] = cs * abs(np.sum(np.multiply(a, matrix4)) * 0.25)
        cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Sobel4.bmp',image1)
        return image1
    
    else:
        return image1

def Laplacian(image):
    rows,cols,_ = image.shape
    image1 = image*0
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    for i in range(1, rows-1):
        for j in range(1, cols-1):
            a = np.array([(grayImg[i-1,j-1],grayImg[i-1,j],grayImg[i-1,j+1]),
                        (grayImg[i,j-1],  grayImg[i,j],  grayImg[i,j+1]),
                        (grayImg[i+1,j-1],grayImg[i+1,j],grayImg[i+1,j+1])])
            image1[i][j] = cl * abs(np.sum(np.multiply(a, matrix5)) * 0.125)      
    cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/Result/Laplacian.bmp',image1)
    return image1


# cv2.imshow("Simple", cv2.cvtColor(Simple(1,image), cv2.COLOR_BGR2GRAY)/255)
# cv2.imshow("Sobel", cv2.cvtColor(Sobel(4,image), cv2.COLOR_BGR2GRAY)/255)
cv2.imshow("Laplacian", cv2.cvtColor(Laplacian(image), cv2.COLOR_BGR2GRAY)/255)
cv2.waitKey(0)
cv2.destroyAllWindows()