#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from os.path import basename
from glob import glob
from imutils import perspective
from imutils import contours
import numpy as np
import imutils

def get_contours(img):
    # First make the image 1-bit and get contours
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

    img2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # filter contours that are too large or small
    size = get_size(img)
    contours = [cc for cc in contours if contourOK(cc, size)]
    return contours

def get_size(img):
    ih, iw = img.shape[:2]
    return iw * ih

def contourOK(cc, size=1000000):
    x, y, w, h = cv2.boundingRect(cc)
    if w < 50 or h < 50: return False # too narrow or wide is bad
    area = cv2.contourArea(cc)
    return area < (size * 0.25) and area > 4000


# In[2]:


def find_4_coord(contours):
    for cc in contours:
        x, y, w, h = cv2.boundingRect(cc)

        approx = cv2.approxPolyDP(cc, 0.09 * cv2.arcLength(cc, True), True)
        n = approx.ravel() 
        i = 0
        box = []
        for j in n : 
            if(i % 2 == 0): 
                box.append([n[i], n[i + 1]]) 
            i = i + 1
    return np.array(box)
        
def order_points(pts):

    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transform(img, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    print(tl, tr, br, bl)

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


# In[16]:


img = cv2.imread('ktp5.jpg')

contours = get_contours(img)
pts = find_4_coord(contours)

img_transform = transform(img, pts)


cv2.imshow("Croped Photo", img_transform)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[34]:





# In[ ]:




