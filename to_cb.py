import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2

size=576

model=tf.keras.applications.resnet50.ResNet50(include_top=True,
                                              input_shape=(size, size, 3),
                                              pooling=None,
                                              weights=None,#'imagenet',
                                              classes=102)

model=tf.keras.models.load_model('D:/!!ufa/skfo/mdl_wts65.hdf5')

model_ex = keras.Model(inputs=model.inputs,
                       outputs=model.get_layer(name="avg_pool").output)



from tqdm import tqdm

arr=[]
for i in tqdm(os.listdir('D:\!!ufa\skfo\sum')):
    for j in os.listdir(f'D:/!!ufa/skfo/sum/{i}'):
        imgdir=f'D:/!!ufa/skfo/sum/{i}/{j}'
        img=cv2.imread(imgdir)
        img=img/255
        img=cv2.resize(img, (576, 576))
        img=img.reshape(1, 576, 576, 3)
        x = tf.expand_dims(img, axis=0)
        y = model_ex(x[0])
        y=y[0].numpy()
        y=np.append(y, i)
        arr.append(y)

df=pd.DataFrame(arr)
print(df)
df.to_csv('tocb.csv', index=False)
'''for i in tqdm(train_generator):
    x = tf.expand_dims(train_generator[0][0], axis=0)
    y = model_ex(x[0])
    arr.append(list(y[0]))'''
