from re import I
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

image = cv.imread('Placa.png')
ig = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.imshow(cv.cvtColor(ig, cv.COLOR_BGR2RGB))
i_bf = cv.bilateralFilter(ig, 11, 17, 17) 
i_edged = cv.Canny(i_bf, 30, 200) 
plt.imshow(cv.cvtColor(i_edged, cv.COLOR_BGR2RGB))

i_kp = cv.findContours(i_edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
ic = imutils.grab_contours(i_kp)
ic = sorted(ic, key=cv.contourArea, reverse=True)[:10]

location = None
for contour in ic:
    approx = cv.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(ig.shape, np.uint8)
new_image = cv.drawContours(mask, [location], 0,255, -1)
new_image = cv.bitwise_and(image, image, mask=mask)

plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = ig[x1:x2+1, y1:y2+1]
plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

text = result[0][-2]
font = cv.FONT_HERSHEY_SIMPLEX
res = cv.putText(image, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))