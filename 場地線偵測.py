
import numpy as np
import cv2
import glob as gb
params = []
params.append(cv2.IMWRITE_PNG_COMPRESSION)
line_o=np.zeros(shape=(1,4))
index=[0]
count=0
#input
img_path = gb.glob("input\\200_0.png")
img_path.sort(key=len)
for path in img_path:
    src  = cv2.imread(path)
    file_name=path.split("\\")
    #轉color base
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    #去除觀眾
    ad = cv2.inRange(hsv, (20,0,63),(101,255,255))
    #開運算
    kernel = np.ones((5,5), np.uint8)
    ad = cv2.morphologyEx(ad,cv2.MORPH_OPEN, kernel)
    contours,_= cv2.findContours(ad,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    s=src.copy()#複製
    
    #凸包(Convex Hull)
    big_contours=[]
    for cnt in contours:
        a=[len(cnt)]
        big_contours.extend(a)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if len(cnt)>np.mean(big_contours):
            for i in range(len(hull)):
                cv2.drawContours(s,[hull],0,(255,255,255),-1)
    #轉灰階
    s= cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    #二值化
    ret, th2 = cv2.threshold(s, 254, 255, cv2.THRESH_BINARY)
    #mask
    ad = cv2.bitwise_and(th2,ad)
    masked_gray = cv2.bitwise_and(src,src, mask = ad)
    ad = cv2.cvtColor(masked_gray, cv2.COLOR_BGR2GRAY) 
    #開運算
    Binary = cv2.inRange(hsv,(34,0,0),(79,111,255))
    kernel = np.ones((5,5), np.uint8)
    Binary = cv2.morphologyEx(Binary,cv2.MORPH_OPEN, kernel)
    #TOPHAT去除光線陰影
    kernel = np.ones((9,9), np.uint8)
    result_tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    masked_gray2 = cv2.bitwise_and(result_tophat,result_tophat, mask = Binary)
    masked_gray2=cv2.cvtColor(masked_gray2, cv2.COLOR_BGR2GRAY)
    
    #去雜訊
    kernel = np.array((
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]), dtype="float32")
    dst_x= cv2.filter2D(ad,-1,kernel)
    dst_x2= cv2.filter2D(masked_gray2,-1,kernel)
    
    kernel = np.array((
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]), dtype="float32")
    dst_y= cv2.filter2D(ad,-1,kernel)
    dst_y2= cv2.filter2D(masked_gray2,-1,kernel)
    
    masked_gray = cv2.bitwise_or(dst_x,dst_y)
    masked_gray2 = cv2.bitwise_or(dst_x2,dst_y2)
    
    #TOPHAT去除光線陰影
    masked_gray = cv2.bitwise_or(masked_gray,masked_gray2)
    kernel = np.ones((9,9), np.uint8)
    masked_gray = cv2.morphologyEx(masked_gray, cv2.MORPH_TOPHAT, kernel)
    
    for i in range(0,720):
        if (ad[i,0:1280].sum()/1280<220):#跳過觀眾席的pixel
            for j in range(0,1280):
                if  (masked_gray[i,j]>30) and ad[i,j]>0:
                    src[i,j]=(0,0,255)#紅色
    cv2.imwrite("output\\"+file_name[-1], src, params)
