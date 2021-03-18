import tensorflow.keras as keras
import data_extraction
import numpy as np
import cv2
import imageio
import scipy
import tensorflow as tf
import math


print(" jer")
model_1=keras.models.load_model('./testing_model')
xs = []
ys = []
smoothed_angle = 0

img = cv2.imread("steering wheel.jpg")

rows,cols = img.shape[0],img.shape[1]

data = "./driving_dataset/data.txt"
with open(data) as f:
    for line in f:
        xs.append(line.split()[0])
        ys.append(float(line.split()[1])*(scipy.pi/180));

num_images = len(xs)

i = math.ceil(num_images*0.8)
print("Starting frameofvideo:" +str(i))

while(cv2.waitKey(10) != ord('q')):
    full_image = imageio.imread("driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200,66)) / 255.0
    image = tf.expand_dims(image,axis=0)
    degrees = model_1.predict(np.array(image))[0][0] * 180.0 / scipy.pi

    print("Steering angle: " + str(degrees) + " (pred)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
