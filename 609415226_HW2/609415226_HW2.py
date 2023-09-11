import numpy as np
import cv2

# RGB2XYZ空間係數矩陣
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])

#   這是將RGB彩色img轉為HSI img的函数
def RGB2HSI(rgb_img):
  
    #保存orginal img的行列数
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    #對原始圖像進行複製
    hsi_img = rgb_img.copy()
    #對圖像進行通道拆分
    B,G,R = cv2.split(rgb_img)
    #把通道化為[0,1]
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col))    #定義H通道
    I = (R + G + B) / 3.0       #計算I通道
    S = np.zeros((row,col))     #定義S通道
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   #計算夾角
        h = np.zeros(col)       #定義臨時數組
        #den>0且G>=B的元素h赋值為thetha
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        #den>0且G<=B的元素h赋值為thetha
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        #den<0的元素h賦值為0
        h[den == 0] = 0
        H[i] = h/(2*np.pi)      #弧度化後赋值给H通道
    #計算S通道
    for i in range(row):
        min = []
        #找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        #計算S通道
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        #I為0的值直接 = 0
        S[i][R[i]+B[i]+G[i] == 0] = 0
    #255以方便顯示，一般H分量在[0,2pi]之间，S和I在[0,1]之间   
    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    hsi_img[:,:,2]= np.power(hsi_img[:,:,2]/255,2.5)*255
    return hsi_img


def HSI2RGB(hsi_img):
    
    #這是將HSI_img轉為RGB_img的函数
    
    # 保存原始img的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    #對原始img進行複製
    rgb_img = hsi_img.copy()
    #對img進行通道拆分
    H,S,I = cv2.split(hsi_img)
    #把通道正規化到[0,1]
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]   
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
#       H>= 0 <120  第一種情況
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
#       H>= 120 <240 第二種情況
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
#       H>= 240 <360 第三種情況
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2             
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255    
    return rgb_img

# channel取值范围：[0,1]
def f(channel):
    return np.power(channel, 1 / 3) if channel > 0.008856 else 7.787 * channel + 0.137931


def anti_f(channel):
    if channel > 0.206893 :
        return np.power(channel, 3)
    else:
        return (channel - 0.137931) / 7.787

#  RGB to Lab
# 像素值RGB轉XYZ空間，pixel格式:(B,G,R)
# 返回XYZ空間下的值
def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    #rgb = rgb / 255.0
  
    XYZ = np.dot(M, rgb)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    """
    XYZ空间转Lab空间
    :param xyz: 像素xyz空间下的值
    :return: 返回Lab空间下的值
    """
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    """
    RGB空间转Lab空间
    :param pixel: RGB空间像素值，格式：[G,B,R]
    :return: 返回Lab空间下的值
    """
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    
    return Lab

# Lab 轉 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb


def run_main():    
#   這是主函數 
    rgb_img = cv2.imread('house.jpg')
    size = rgb_img.shape
#   創建空間
    img_new = np.zeros(size)
    lab = np.zeros(size)
#   進行轉換    
    hsi_img = RGB2HSI(rgb_img)       
    rgb_img2 = HSI2RGB(hsi_img)   
    for i in range(size[0]):
        for j in range(size[1]):
            Lab = RGB2Lab(rgb_img[i,j])
            lab[i, j] = (Lab[0], Lab[1], Lab[2])
    lab = np.power(lab/255,2.5)*255   
    for i in range(size[0]):
        for j in range(size[1]):
            rgb = Lab2RGB(lab[i,j])
            img_new[i, j] = (rgb[2], rgb[1], rgb[0])
    cv2.imshow("Origin",rgb_img)
    cv2.imshow("HSI", hsi_img)
    cv2.imshow("HSI2RGB",rgb_img2.astype('uint8'))
    cv2.imshow("LAB",lab)
    cv2.imshow("LAB2RGB",img_new.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_main()