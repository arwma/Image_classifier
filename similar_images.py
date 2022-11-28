import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
input_dir = '/kaggle/input/fashion-product-images-small/images'
glob_dir = input_dir + '/*.jpg'

temp  = glob.glob(glob_dir)

print(len(temp))
x=temp[0:1000]
paths = [file for file in x]
images = [cv2.resize(cv2.imread(file), (128, 128)) for file in x]
images = np.array(np.float32(images).reshape(len(images),images[0].shape[0],images[0].shape[1],images[0].shape[2])/255)
images.shape
plt.imshow(images[0])
## embedding
model = tf.keras.applications.MobileNetV2(include_top = False, weights='imagenet', input_shape=(128, 128, 3))


predictions = model.predict(images.reshape(-1, 128, 128, 3)) ## representation of image to reduce size



pred_images = predictions.reshape(images.shape[0], -1) ## reshape
pred_images.shape
#### Plotting Silhoutte score to find optimal K
sil = []
kl = []
kmax = 10



for k in range(2, kmax+1):
  print(k)
  kmeans2 = KMeans(n_clusters = k,random_state = 0).fit(pred_images)
  labels = kmeans2.labels_
  sil.append(silhouette_score(pred_images, labels))
  kl.append(k)
  plt.plot(kl, sil)
plt.ylabel('Silhoutte Score')
plt.ylabel('K')
plt.show()
Training K-means
Here maximum is at k = 3 . But as this is fashion dataset so there can be many different types of clusters like based on gender , top or bottom etc. So, for this dataset k = 3 don't seems to be sufficient so choosing k = 8

k = 8
kmodel = KMeans(n_clusters=k, n_jobs=-1, random_state = 0) #change random state any value

kmodel.fit(pred_images) ### training kmean


kpredictions = kmodel.predict(pred_images) ###prediction
####
#### Making output directory
for i in range(k):
    if(os.path.isdir("/kaggle/working/cluster" + str(i))==False):
        os.makedirs("/kaggle/working/cluster" + str(i))
    #print(i)
    ## for making directory
    
    
    ## Saving images in directory
for i in range(len(paths)):
    shutil.copy2(paths[i], "/kaggle/working/cluster"+str(kpredictions[i]))
####
#### Plotting image from each cluster
fig ,axs = plt.subplots(k,5,figsize=(50,50)) ### making subplot 

for i in range(k):
    xyz = '/kaggle/working/cluster'+str(i) + '/*.jpg'
    img = [cv2.imread(abc,0) for abc in glob.glob(xyz)]
    
    
    #print("Cluster No "+str(i))
    for j in range(min(5, len(img))):
      axs[i][j].imshow(img[j], cmap = 'gray', interpolation = 'bicubic')
      axs[i][j].set_xticks([])
      axs[i][j].set_yticks([])
####
