import re
import os
import json
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
from keras.preprocessing.image import img_to_array


# Download caption annotation files 20GB files took around 20 mins to run


annotation_folder = '../data/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                          extract = True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'
  os.remove(annotation_zip)

# Download image files
image_folder = '../train2017/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('train2017.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2017.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder

  # Paths
captions_path = '../data/annotations/captions_train2017.json'
annotations_path = '../data/annotations/instances_train2017.json'
images_path = '../data/train2017/'

dir_content = os.listdir(images_path)

# Load captions
with open(captions_path) as json_file: 
    captions = json.load(json_file)

# Load anotations
with open(annotations_path) as json_file: 
    annotations = json.load(json_file)


# Read non bounded image meta


#Reading dataframes
df_images = pd.read_csv('Unique_images_with_Labels.csv',index_col = 0)
df_train = pd.read_csv('captions_train.csv',names=['Images','Text'])
df_test = pd.read_csv('captions_test.csv',names=['Images','Text'])


# Define class mappings


dict_map = {'vehicle':3,'plane':5,'cat':17,'dog':18}
def encoder(x):
  return dict_map[x.lower()]

df_images['Encoded_Labels'] = df_images['Label'].apply(lambda x: encoder(x))


# Filter images and BBs for the classes


images_filenames = df_images['Images'].iloc[1:,:].to_list()
encoded_labels = df_images['Encoded_Labels'].to_list()
int_categories = [3,5,17,18]

bbox = []
for idx,names in enumerate(images_filenames):
  name = names.split('.')[0]
  image_id = re.sub('^0+','',name)
  image_id = int(image_id)
  category = encoded_labels[idx]
  for i in annotations['annotations']:
    if(i['image_id'] == image_id and i['category_id']==category):
      bbox.append([image_id,i['bbox'],names,category])
      break


# Apply bounding boxes


os.chdir('/content/crop_images')

for box in bbox:
  img_name = box[2]
  xmin,ymin,width,height = box[1][0],box[1][1],box[1][2],box[1][3]
  xmax = xmin + width
  ymax = ymin + height
  area = (xmin,ymin,xmax,ymax)
  img_path = os.path.join(images_path,img_name)
  img = Image.open(img_path)
  im =img.crop(area)
  im = img_to_array(im)
  cv2.imwrite(img_name,im)


# Create meta


images_retained = []
for i in os.listdir('/content/crop_images'):
  images_retained.append(i)

mask = df_images['Images'].apply(lambda x: x in images_retained)
mask1 = df_train['Images_name'].apply(lambda x: x in images_retained)
mask2 = df_test['Images_name'].apply(lambda x: x in images_retained)

df_modified = df_images[mask]
df_train_modified = df_train[mask1]
df_test_modified = df_test[mask2]


# Save meta


df_train_modified.to_csv('../data/crop_images/captions_train.csv',header=True)
df_test_modified.to_csv('../data/crop_images/captions_test.csv',header=True)
df_modified.to_csv('../data/crop_images/unique_images_with_labels.csv',header=True)

