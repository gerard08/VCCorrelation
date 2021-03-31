import cv2
import numpy as np
from scipy import signal
from scipy import misc
from scipy.ndimage.interpolation import shift
import os.path
import time

from matplotlib import pyplot as plt


# import skimage
# shape (640, 247, 3)
def Crop(imagesrc):
    channels = []
    img = cv2.imread(imagesrc, cv2.IMREAD_GRAYSCALE)

    for h in range(0, int(img.shape[0] / 3) * 3, int(img.shape[0] / 3)):
        channels.append(img[h:h + int(img.shape[0] / 3), 0:img.shape[1]])
    return channels


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def calculate_displacement(channel_A, channel_B, method, fourier):
    crop_size_x = int(channel_B.shape[1] * 0.8)
    crop_size_y = int(channel_B.shape[0] * 0.8)

    mascara = crop_center(channel_B, crop_size_x, crop_size_y)

    if not fourier:
        correlation = cv2.matchTemplate(channel_A, mascara, method)
    else:
        correlation = fourierCorrelation(channel_A, mascara)
        
    max_correlation = np.unravel_index(np.argmax(correlation), correlation.shape)
    # print(max_correlation)
    center_correlation = (correlation.shape[0] // 2, correlation.shape[1] // 2)
    # print(center_correlation)
    return np.subtract(max_correlation, center_correlation)


def Fusion_Channels(img_path, method, fourier = False):
    t1 = time.time()
    #Crop the triple image in three different images
    channels = Crop(img_path)
    t2 = time.time()
    print("cutting time:", t2 - t1, "s")


    t1 = time.time()
    # calculate displacement blue-green
    b_g_displacement = calculate_displacement(channels[0], channels[1], method, fourier)
    # calculate displacement blue-red
    b_r_displacement = calculate_displacement(channels[0], channels[2], method, fourier)
    #Displace the channels
    displaced_channel_g = shift(channels[1], b_g_displacement, cval=0)
    displaced_channel_r = shift(channels[2], b_r_displacement, cval=0)
    t2 = time.time()
    print("displacement calculation time:", t2 - t1, "s")

    t1 = time.time()
    # Fusion the three channels BGR
    color_img_result = np.dstack([channels[0], displaced_channel_g, displaced_channel_r])

    cv2.imshow('color_img', color_img_result)
    t2 = time.time()
    print("fusion time:", t2 - t1, "s")
    cv2.waitKey(0)

    img_name = os.path.basename(img_path)
    root, ext = os.path.splitext(img_name)
    t1 = time.time()
    cv2.imwrite(os.path.join("result", root + "_color" + ext), color_img_result)
    t2 = time.time()
    print("writing time:", t2 - t1, "s")

def fourierCorrelation(img, mask):

    #fem la transformada de la img i la mascara
     f = np.fft.fft2(img)
     m = np.fft.fft2(mask)#.conj()

    #arreglem problemes amb els tamanys de la mascara
     padx = (f.shape[0] - m.shape[1]) // 2
     pady = (f.shape[1] - m.shape[0]) // 2

     if(m.shape[0] + (pady * 2)) == f.shape[1]:
         nm = np.pad(m, [(pady, pady), (padx, padx)], 'minimum')#.conj()
     else:
         nm = np.pad(m, [(pady, pady + 1), (padx, padx)], 'minimum')#.conj()

     # faig el producte de la imatge amb la mascara
     correlation = np.matmul(f, nm)

     result_img = np.fft.ifft2(correlation)
     result_img = np.abs(result_img) #Remove complex values

     ncor = result_img

     plt.subplot(122),plt.imshow(ncor, cmap = 'gray')
     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
     plt.show()
     return ncor



if __name__ == '__main__':
    # Especify the correlation method
    # 0 = TM_SQDIFF
    # 1 = TM_SQDIFF_NORMED
    # 2 = TM_CCORR
    # 3 = TM_CCORR_NORMED
    # 4 = TM_CCOEFF
    # 5 = TM_CCOEFF_NORMED
    method = cv2.TM_CCOEFF_NORMED

    #Fusion the channels for every triple image inside the directory
    for img_path in os.listdir("Img_triple"):
        print("Image Name: ",img_path)
        img_path = os.path.join("Img_triple", img_path)
        Fusion_Channels(img_path, method, True  )
        print("/////////////////////////")
