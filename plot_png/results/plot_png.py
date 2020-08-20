import cv2
import numpy as np
from PIL import Image

imagename = 'slime'

img0 = cv2.imread('plot_png/' + imagename + '0.png', cv2.IMREAD_UNCHANGED)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2RGBA)
imgfinal = img0

for i in range(1, 279):
    img0 = cv2.imread('plot_png/' + imagename + str(i) + '.png', cv2.IMREAD_UNCHANGED)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2RGBA)

    imgfinal = cv2.addWeighted(imgfinal,0.8,img0,0.8,0)

imgfinali = Image.fromarray(imgfinal)
imgfinali.save('slime.png')