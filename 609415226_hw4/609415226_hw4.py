import numpy as np
import cv2

#Gaussian filter 3*3
sigma = 1
x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp(-(x**2+y**2) / (2*sigma**2))
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

#Laplacian kernel
Laplacian_kernel  = np.array([[ 0,-1, 0],
                              [-1, 5,-1],
                              [ 0,-1, 0]])

def Gaussian_blur(inputimg):
    Gaussian = np.zeros(inputimg.shape)
    for i in range(1, inputimg.shape[0]-1):
        for j in range(1, inputimg.shape[1]-1):
             Gaussian[i-1, j-1] = np.sum(gaussian_kernel * inputimg[i-1:i+2, j-1:j+2])             
    return Gaussian

#a function for finding X gradient
def SobelXgradient(image):
    size = image.shape
    output = np.zeros(size)
    Gx = np.array(np.mat('1 0 -1; 2 0 -2; 1 0 -1'))
    for i in range(1, size[0]-1):
         for j in range(1, size[1]-1):
             output[i-1][j-1] = np.sum(Gx*image[i-1:i+2,j-1:j+2])
    return output

#a function for finding Y gradient
def SobelYgradient(image):
    size = image.shape
    output = np.zeros(size)
    Gy = np.array(np.mat('1 2 1; 0 0 0; -1 -2 -1'))
    for i in range(1, size[0]-1):
         for j in range(1, size[1]-1):
             output[i-1][j-1] = np.sum(Gy*image[i-1:i+2,j-1:j+2])
    return output

def Laplacian(image):
    size = image.shape
    z = np.zeros(size)
    
    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):
            z[i-1][j-1] = np.sum(Laplacian_kernel * image[i-1:i+2, j-1:j+2])
    return z

    #reading an image 
original = cv2.imread('image3.jpg',0)
    
Gaussian = Gaussian_blur(original)
    
sobelx = SobelXgradient(Gaussian)
sobely = SobelYgradient(Gaussian)
sobel = np.sqrt(sobelx**2 + sobely**2)
sobel = sobel / sobel.max() * 255 
Laplacian_img  = Laplacian(Gaussian)
cv2.imshow('og_image',original)
cv2.imshow('Sobel',sobel.astype('uint8'))
cv2.imshow('Log',Laplacian_img.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
    
   