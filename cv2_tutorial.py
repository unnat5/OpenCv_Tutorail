import cv2 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('dp.jpg',cv2.IMREAD_GRAYSCALE)

# cv2.imshow('image',img)
# cv2.waitKey(0) #wait for any key to be pressed
# cv2.destroyAllWindows()


############using matplotmib#####
# plt.imshow(img,cmap = 'gray',interpolation='bicubic')
# plt.plot([50,100],[80,100],'c',linewidth=5)
# plt.show()


#### capturing video####
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

##########################
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow("frame1",gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


##############################
# Part 3
#img = cv2.imread("dp.jpg",cv2.IMREAD_COLOR)
#cv2.line(img,(0,0),(150,150),(255,255,255),15) 
#(input,starting_point,ending_point,(color_code),line_width)
#BGR
#blue = (255,0,0)
#green = (0,255,0)
#red = (0,0,255)
#black= (0,0,0)
#white= (255,255,255)
# cv2.rectangle(img,(15,25),(200,150),(0,255,0),5)
# #                   _topleft,bottomright_
# # negative linewidth will fill in the shape
# cv2.circle(img,(300,150),89,(0,0,255),-1)
# pts  = np.array([[10,29],[232,23],[443,12],[53,421]],np.int32)
# cv2.polylines(img,[pts],True,(0,255,255),5)
# #                     True:whether we want to close the shape

# font=cv2.FONT_HERSHEY_COMPLEX
# cv2.putText(img,'OpenCv pUTS!',(0,330),font,1,(0,0,0),2,cv2.LINE_AA)
# #(img,text,loc,font,size,color,thickness,)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#################################################
# img = cv2.imread('dp.jpg',cv2.IMREAD_COLOR)
# #ROI - REGION OF IMAGE
# face=img[93:422,121:372] 
# img[0:329,0:251] = face
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


################################################
#Image arithmatic
#img1 = cv2.imread('3D-Matplotlib.png')
#img2 = cv2.imread('mainsvmimage.png')
#img_sum = img1+img2

#img_sum=cv2.add(img1,img2)
## this will add all the pixels and resultant image will be bright

#weighted = cv2.addWeighted(img1,0.6,img2,0.4,0)
##last value is gamma
# cv2.imshow('figure',weighted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #####################################################
# img1 = cv2.imread('3D-Matplotlib.png')
# #img2 = cv2.imread('mainsvmimage.png')
# img2 = cv2.imread('mainlogo.png')

# rows,cols,channels = img2.shape
# roi = img1[0:rows,0:cols]

# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# ret,mask = cv2.threshold(img2gray,220,225,cv2.THRESH_BINARY_INV)
# # SO here if any pixel is greater than 220 then it is coverted to 255 and
# #if it is below 220 then converted to black than it inverses it.
# mask_inv1 = cv2.bitwise_not(mask)

# img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv1)
# #img1_bg = cv2.bitwise_not(img1_bg)
# img2_fg = cv2.bitwise_and(img2,img2,mask=mask)

# dst = img1_bg+img2_fg


# img1[0:rows,0:cols] = dst
# #print(mask)
# cv2.imshow('dst',dst)
# cv2.imshow('mask',mask)
# cv2.imshow('mask_inv',mask_inv1)
# cv2.imshow('img1_bg',img1_bg)
# cv2.imshow('img2_fg',img2_fg)
# cv2.imshow('img1',img1)
# cv2.waitKey()
# cv2.destroyAllWindows()




#################################################
# import cv2
# import numpy as np

# # Load two images
# img1 = cv2.imread('3D-Matplotlib.png')
# img2 = cv2.imread('mainlogo.png')

# # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols ]

# # Now create a mask of logo and create its inverse mask
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# # add a threshold
# ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

# mask_inv = cv2.bitwise_not(mask)

# # Now black-out the area of logo in ROI
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# # Take only region of logo from logo image.
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst

# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#########################


# #Image arithmatic
# img1 = cv2.imread('3D-Matplotlib.png')
# #img2 = cv2.imread('mainsvmimage.png')
# img2 = cv2.imread('mainlogo.png')
# img3 = cv2.imread('PYTHON.jpg')
# img3 = cv2.resize(img3,(128,128))
# #img_sum = img1+img2
# #img_sum = cv2.add(img1,img2)
# #weighted = cv2.addWeighted(img1,0.6,img2,0.4,0)

# #cv2.imshow('figure',weighted)
# #print(img2)
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols ]

# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# ret,mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)
# # SO here if any pixel is greater than 220 then it is coverted to 255 and
# #if it is below 220 then converted to black than it inverses it.

# mask_inv = cv2.bitwise_not(mask)

# img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
# img2_fg = cv2.bitwise_and(img2,img2,mask=mask)

# dst = cv2.add(img1_bg,img2_fg)

# img1[0:rows,0:cols] = dst

# cv2.imshow('img',img1)
# cv2.imshow('p',img3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

###################################################




# img1 = cv2.imread('3D-Matplotlib.png')
# #img2 = cv2.imread('mainsvmimage.png')
# img2 = cv2.imread('mainlogo.png')
img3 = cv2.imread('PYTHON.jpg')
# img3 = cv2.resize(img3,(128,128))


# rows,cols,channels = img3.shape
# roi = img1[0:rows,0:cols]
# img3gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

# ret,mask = cv2.threshold(img3gray,10,255,cv2.THRESH_BINARY_INV)
# inv_mask = cv2.bitwise_not(mask)

# im1 = cv2.bitwise_and(roi,roi,mask=mask)
# im2 = cv2.bitwise_and(img3,img3,mask=inv_mask)

# fim = cv2.add(im1,im2)

# img1[0:rows,0:cols] = fim
# cv2.imshow('f',img1)
# cv2.imshow('g',im2)
# cv2.imshow('dumq',inv_mask) #black 
# cv2.imshow('dum',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




###############################################
# img1  = cv2.imread("bookpage.jpg",cv2.IMREAD_COLOR)
# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret,mask = cv2.threshold(img1,10,225,cv2.THRESH_BINARY)
# guass = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              cv2.THRESH_BINARY,115,1)

# retval2 , otsu = cv2.threshold(img2,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('fig',mask)
# cv2.imshow('fig1',guass)
# cv2.imshow('figW',otsu)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




#############################EXPERIMENTING WITH WEED IMAGE ######################
# img1 = cv2.imread('weed1.png')
# img2 = cv2.imread('weed2.png')
# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# tet,guass = cv2.threshold(img1,60,255,cv2.THRESH_BINARY)
# cv2.imshow("fig",img1)
# cv2.imshow('if',guass)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

################################################


# color filtering


# cap = cv2.VideoCapture(0) ## 0 first webcam

# while True:
#     _, frame = cap.read()
#     #hue saturation value
#     # IN HSV all the values are independent of each other
#     # unlike BGR.
#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
#     lower_orange = np.array([10,100,20])
#     upper_orange = np.array([15,255,255])
#     ## where there is white 1(if its in range and 0 otherwise) we
#     #will show that color in mask.
    
#     mask = cv2.inRange(hsv,lower_orange,upper_orange)
    
#     res= cv2.bitwise_and(frame,frame,mask=mask)
#     cv2.imshow("frame",res)
#     cv2.imshow("mask",mask)
#     #cv2.imshow("frame1",frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

   

    

# cv2.destroyAllWindows()
# cap.release()


########################################################
weed1= cv2.imread('weed1.png',cv2.IMREAD_COLOR)
weed_hsv = cv2.cvtColor(weed1,cv2.COLOR_BGR2HSV)
sensitivity = 30
a =np.array([60 - sensitivity, 100, 50])  #lower_Range
b=np.array([60 + sensitivity, 255, 255]) # upper_Range
#lower_green = np.array([60,100,50])
#upper_greeen = np.array([60,255,255])
#lower_orange = np.array([10,100,20])
#upper_orange = np.array([15,255,255])
mask = cv2.inRange(weed_hsv,a,b)

res_w = cv2.bitwise_and(weed1,weed1,mask= mask)

cv2.imshow('fig',res_w)
cv2.imshow('figure',mask)
cv2.imshow('original',weed1)

cv2.waitKey(0)
cv2.destroyAllWindows()















