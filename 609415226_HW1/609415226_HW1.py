import numpy as np
import cv2
import math
name = ['blurry_moon.tif', 'skeleton_orig.bmp']
A = 3

for x in name:
    origin = cv2.imread(x, 0)
    size = origin.shape
    
    fft2 = np.fft.fft2(origin)
    lap = fft2.copy()
    shift2center = np.fft.fftshift(fft2)
    shift2center[int((size[0]/2)-1) : int((size[0]/2)+1), int((size[1]/2)-1) : int((size[1]/2)+1)] = 0
    
    for i in range(size[0]):
        for j in range(size[1]):
            lap[i][j] = -4*(math.pi**2)*abs((i-size[0]/2)**2 + (j-size[1]/2)**2)*shift2center[i][j]
    
    center2shift = np.fft.ifftshift(lap)
    ifft2 = np.fft.ifft2(center2shift)
    lap_img = np.abs(ifft2)/np.max(np.abs(ifft2))
    
    #sharpening
    sharpen = lap_img + (origin/255)
    
    #unsharpening
    unsharp = (origin/255) - lap_img
    
    #high-boost
    highboost = A*(origin/255) - lap_img

    cv2.imshow(("origin_" + x), origin)
    cv2.imshow(('Laplacian operator_' + x), sharpen)
    cv2.imshow(('unsharp masking_' + x), unsharp)
    cv2.imshow((' high-boost filtering_' + x), highboost)
  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    