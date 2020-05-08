# Download caption annotation files 20GB files took around 10 mins to run


annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                          extract = True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'
  os.remove(annotation_zip)

# Download image files

image_folder = '/train2017/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('train2017.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2017.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder
  
# Paths in Google Colab if ran
captions_path = '/content/annotations/captions_train2017.json'
annotations_path = '/content/annotations/instances_train2017.json'
images_path = '/content/train2017/'

dir_content = os.listdir(images_path)
# Load captions
with open(captions_path) as json_file: 
    captions = json.load(json_file)

# Load anotations
with open(annotations_path) as json_file: 
    annotations = json.load(json_file)
