import os
import imageio
import scipy
import scipy.misc
import cv2

pointer_1= 0   # global variables
pointer_2 =13

x = []
y = []

data = "./driving_dataset/data.txt"

with open(data) as f:
    for line in f:
        x.append(line.split()[0])
        y.append(float(line.split()[1])*(scipy.pi/180));

x_train = x[0:int(len(x)*0.8)]
y_train = y[0:int(len(y)*0.8)]

x_test = x[int(len(x)*0.8)+1:]
y_test = y[int(len(y)*0.8)+1:]



def get_test_batch(batch_size):
    global pointer_2
    x_=[]
    y_=[]

    for i in range(batch_size):
        x_.append(cv2.resize(imageio.imread(os.path.join("driving_dataset",x_test[(pointer_2+i)%len(x_test)]))[-150:],(200,66))/255.0)
        y_.append(y_test[(pointer_2+i)%len(y_test)])
    pointer_2 += batch_size

    return x_,y_















