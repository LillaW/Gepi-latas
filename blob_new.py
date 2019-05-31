##############Dobokocka felismerese es pontjainak osszeadasa#############


import numpy as np
import cv2


def ComputeContourCenter(contour): 
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX,cY)
    return center


def ComputeBoxSize(box):
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    size =(x2-x1, y2-y1)
    return size

############Kep kivalasztasa##############
img = cv2.imread("01.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##Kep szerkesztese(szurkites)## 
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)     
ret,thresh = cv2.threshold(gray,0,255,(cv2.THRESH_BINARY+cv2.THRESH_OTSU))

##Konturbeallitasok##
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cnts = sorted(contours,key=cv2.contourArea,reverse=True)

##Pontszamlalo(Ossz) valtozo nullazasa##
sum_dots = 0
for c in cnts:
    ##Kozelito ertekek beallitasa hatarvonalakhoz##
    approx = cv2.approxPolyDP(c,0.04*cv2.arcLength(c,True),True)
    area = cv2.contourArea(c)
    if ((len(approx)==4) & (area > 1000)):
        cv2.drawContours(img,[c],0,(0,255,0),3)
        
        ##Kockamegtalalasi ertekek beallítasa##
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        ##Keresett terulet/kockahatarvonal kijelolese, fgv-ek hasznalata##
        center =ComputeContourCenter(c)
        size = ComputeBoxSize(box)           
        roi = cv2.getRectSubPix(gray,size,center)
        
        ##Pontok megszamlalasanak beallitasai, Blob Detector hasznalata##
        params = cv2.SimpleBlobDetector_Params()
        
        ##Circularity filter hasznalata##
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        ##Inertia filter hasznalata##        
        params.filterByInertia = True
        params.minInertiaRatio = 0.75
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(roi)            
        num_dots = len(keypoints)
        
        ##Eltolas kiszamitasa##
        locationX = int(center[0]+size[0]/2)
        locationY = int(center[1]+size[1]/2)
        location = (locationX,locationY)
        
        ##Pottyok kiiratasa es osszeadasa##
        cv2.putText(img,str(num_dots),location,0,1,(0,255,0), 2, cv2.LINE_AA)
        sum_dots += num_dots
        
##Eredmeny kiiratasa##
text = 'Ossz.: ' + str(sum_dots)             
cv2.putText(img,text,(10,30),0,1,(0,255,0), 4, cv2.LINE_AA)

##Kepmegjelenites##
cv2.imshow("Vegeredmeny",img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

