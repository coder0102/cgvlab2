lab 12
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_path = 'passport-photos-different-people-260nw-431262283.webp'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imwrite('detected_faces.jpg', image) 
cv2.imshow('Detected Faces', image) c
cv2.waitKey(0)

lab-11
import cv2
import numpy as np
img=cv2.imread('face.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

lab-10
import cv2
import numpy as np
img=cv2.imread('rabbit.jpg')
def  smooth(img,size):
    return cv2.GaussianBlur(img,(size,size),0)
def blur(img,size):
    return cv2.medianBlur(img,size)
smooth_size=int(input('enter smooth size in odd'))
blur_size=int(input('enter blur size in odd'))
sr=smooth(img,smooth_size)
br=blur(img,blur_size)
cv2.imshow('original',img)
cv2.imshow("smoothed",sr)
cv2.imshow("blured",br)
cv2.waitKey(0)
cv2.destroyAllWindows()

lab-9
import cv2
import numpy as np
img=cv2.imread('rabbit.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,100,200)
filter = np.ones((5, 5), np.float32) / 25 
texture = cv2.filter2D(gray, -1, filter) 
cv2.imshow('original',img)
cv2.imshow('black&white',gray)
cv2.imshow('edges',edge)
cv2.imshow('textured',texture)
cv2.waitKey(0)
cv2.destroyAllWindows()

lab8
import numpy as np
import cv2
img=cv2.imread('rabbit.jpg')
tx=int(input("enter the translational factor along x"))
ty=int(input("enter the translational factor along y"))
fx=float(input("enter the scaling factor along x"))
fy=float(input("enter the scaling factor along y"))
angle=float(input("enter the angle to ratate"))
def translate(img,x,y):
    row,col=img.shape[:2]
    matrix=np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(img,matrix,(row,col))
def scale(img,fx,fy):
    return cv2.resize(img,None,fx=fx,fy=fy)
def rotate(img,angle):
    h,w=img.shape[:2]
    center=(w//2,h//2)
    matrix=cv2.getRotationMatrix2D(center,angle,1)
    return cv2.warpAffine(img,matrix,(w,h))
ti=translate(img,tx,ty)
si=scale(img,fx,fy)
ri=rotate(img, angle)
cv2.imshow('og',img)
cv2.imshow("t",ti)
cv2.imshow("s",si)
cv2.imshow("r",ri)
cv2.waitKey(0)
cv2.destroyAllWindows()

lab 7
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("rabbit.jpg")
h, w = img.shape[:2]

# Split the image into 4 parts
q1, q2 = img[:h//2, :w//2], img[:h//2, w//2:]
q3, q4 = img[h//2:, :w//2], img[h//2:, w//2:]

# Display quadrants
plt.subplot(2, 2, 1), plt.imshow(q1), plt.title("1"), plt.axis("off")
plt.subplot(2, 2, 2), plt.imshow(q2), plt.title("2"), plt.axis("off")
plt.subplot(2, 2, 3), plt.imshow(q3), plt.title("3"), plt.axis("off")
plt.subplot(2, 2, 4), plt.imshow(q4), plt.title("4"), plt.axis("off")

plt.show()
