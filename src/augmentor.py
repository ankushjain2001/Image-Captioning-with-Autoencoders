from numpy import expand_dims
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import numpy as np
import random

def generate_perImageCount(n,target,low,high):
  count_img = [0]*n
  if( n > target):
    indexes = np.random.choice(n,target)
    #print(indexes)
    for idx in indexes:
      count_img[idx] = 1
  else:
    while not sum(count_img) >= target:
      for idx,val in enumerate(count_img):
        x = np.random.randint(low, high)
        if(low == high):
          count_img[idx] +=1
        else:
          count_img[idx] += x
  return count_img

  

def augmentation(type_,image):
  data = img_to_array(image)
  samples = expand_dims(data, 0)

  if(type == 'rotation'):
    rotation_random = np.random.randint(25,90)
    datagen = ImageDataGenerator(type_= rotation_random)
    
  elif(type_ == 'zoom'):
    datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
  else:
    datagen = ImageDataGenerator(horizontal_flip=True)

  it = datagen.flow(samples, batch_size=1)
  batch = it.next()
  image = batch[0].astype('uint8')

    
  return image
  
  
  
def imageAugmentor(x_train, y_train, aug_Images=2500, maintain_Ratio=False):
  x_train_aug = []
  y_train_aug = []
  
  augmentor = ['rotate','zoom','flip']
  dict_class = {}
  total_Count = 0

  print('X Train Size: ',len(x_train))
  print('Y Train Size: ', len(y_train))

  if not maintain_Ratio:
    for cat in set(y_train):
      pres_count = y_train.count(cat)
      dict_class[cat] = (aug_Images - pres_count , pres_count)
  else:
    for cat in set(y_train):
      pres_count = y_train.count(cat)
      dict_class[cat] = (int(aug_Images * (pres_count / len(y_train))), pres_count)

      
  for k,v in dict_class.items():
    print(k,v[0],v[1])

  for clss,img_count in dict_class.items():
    maxImg = int(math.ceil(img_count[0]/img_count[1]))
    generator = generate_perImageCount(img_count[1],img_count[0],1,maxImg)
    index_class = [i for i,category in enumerate(y_train) if category == clss]
    for i,idx in enumerate(index_class):
      image = x_train[idx]
      #For each image how many different augmented images to be formed is stored in perImg
      perImgCount = generator[i]

      while (perImgCount != 0):

        #selection random augmentor type
        index = random.randint(0,len(augmentor)-1)
        augmentor_type = augmentor[index]

        #pass the type and image to augment it
        aug_img = augmentation(augmentor_type,image)
        
        #save the image as img_name#1.jpg  
        x_train_aug.append(aug_img)
        y_train_aug.append(clss)
        
        total_Count +=1
        perImgCount -=1
  return x_train_aug, y_train_aug