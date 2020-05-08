import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model

#gpu_options = tf.GPUOptions(allow_growth=True)
#session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# --- PARAMETERS --------------------------------------------------------------

image_dims = (200, 200, 3) #Normalized image size
n_epochs = 100
batch_size = 64
loss="binary_crossentropy"
optimizer="adam"
# --- DEFINING THE ARCHITECTURE -----------------------------------------------


inputLayer = Input(shape = image_dims)
e1 = Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding='same')(inputLayer) # Padding=same important to maintain reconstrction
e2 = MaxPooling2D(pool_size = (2, 2))(e1)
e3 = Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding='same')(e2)
e4 = MaxPooling2D(pool_size = (2, 2))(e3)
e5 = Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding='same')(e4)
e6 = MaxPooling2D(pool_size = (2, 2))(e5)

d1 = Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding='same')(e6)
d2 = UpSampling2D(size = (2, 2))(d1)
d3 = Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding='same')(d2)
d4 = UpSampling2D(size = (2, 2))(d3)
d5 = Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding='same')(d4)
d6 = UpSampling2D(size = (2, 2))(d5)
outLayer = Conv2D(filters = image_dims[2], kernel_size = (3, 3), activation = 'sigmoid', padding='same')(d6)


# CREATING AUTOENCODER MODEL INSTANCE
# Autoencoder Model
autoencoder = Model(inputLayer, outLayer)

# Encoder Model
encoder = Model(inputLayer, e6)

input_encoder_shape = encoder.layers[0].input_shape[1:]
output_encoder_shape = encoder.layers[-1].output_shape[1:]

# Decoder Model - For producing regenerated images from the autoencoder features
decoded_input = Input(shape=output_encoder_shape)
decoded_output = autoencoder.layers[-7](decoded_input)  # Conv2D
decoded_output = autoencoder.layers[-6](decoded_output)  # UpSampling2D
decoded_output = autoencoder.layers[-5](decoded_output)  # Conv2D
decoded_output = autoencoder.layers[-4](decoded_output)  # UpSampling2D
decoded_output = autoencoder.layers[-3](decoded_output)  # Conv2D
decoded_output = autoencoder.layers[-2](decoded_output)  # UpSampling2D
decoded_output = autoencoder.layers[-1](decoded_output)  # Conv2D

decoder = Model(decoded_input, decoded_output)


#autoencoder.summary()
#encoder.summary()
#decoder.summary()

# -----------------------------------------------------------------------------

shape_img_resize = image_dims
input_shape_model = tuple([int(x) for x in encoder.input.shape[1:]]) #=(200 200 3)
output_shape_model = tuple([int(x) for x in encoder.output.shape[1:]]) #=(26 26 8)




print("input_shape_model = {}".format(input_shape_model))
print("output_shape_model = {}".format(output_shape_model))






#######################################
# IMAGE PREPROCESSING
####################################

path_train = 'dataset500/train/'
path_test = 'dataset500/test/'
index_train = os.listdir(path_train)
index_test = os.listdir(path_test)


# --- LABELS ----
import csv
labels_train_raw=[]
labels_train=[]
labels_test_raw=[]
labels_test=[]

with open('dataset500/meta_train.csv', 'rU') as f:  #opens PW file
    reader = csv.reader(f)
    # Print every value of every row. 
    for row in reader:
        labels_train_raw.append(row)
with open('dataset500/meta_test.csv', 'rU') as f:  #opens PW file
    reader = csv.reader(f)
    # Print every value of every row. 
    for row in reader:
        labels_test_raw.append(row)
            
for i in index_train:
    gotIt=False
    for j in labels_train_raw:
        if i==j[0] and not gotIt:
            gotIt=True
            labels_train.append(j[1])    
for i in index_test:
    gotIt=False
    for j in labels_test_raw:
        if i==j[0] and not gotIt:
            gotIt=True
            labels_test.append(j[1])    
del labels_test_raw
del labels_train_raw
    
# ----------------

def imagePreProcessor(X):
    out_X = []
    for img in tqdm(X):
        # Resize image
        img_new = cv2.resize(img, image_dims[:2])
        # Normalize 8 bit image
        img_new = img_new/255
        # Add image to list
        out_X.append(img_new)
    out_X = np.array(out_X).reshape((-1,) + image_dims)
    return out_X


# Importing train images
X_train_raw = []
for i in tqdm(index_train):
    X_train_raw.append(cv2.imread(path_train+i, 1)) # 1 = read as RGB
# Importing test images
X_test_raw = []
for i in tqdm(index_test):
    X_test_raw.append(cv2.imread(path_test+i, 1)) # 1 = read as RGB


# Preprocessing train images
X_train = imagePreProcessor(X_train_raw)
X_test = imagePreProcessor(X_test_raw)



############################################
# TEXT PREPROCESSING
############################################

path_caps_train = 'datatset500/captions_train.csv'
path_caps_test = 'dataset500/captions_test.csv'









#############################################
# Training the model
autoencoder.compile(loss=loss, optimizer=optimizer)

t0 = time.time()

autoencoder.fit(x=X_train, y=X_train,
                             epochs = n_epochs,
                             batch_size = batch_size,
                             shuffle = True,
                             validation_split=0.1)

t99 = time.time()
print('No GPU Time Taken = ', (t99-t0)/60)
#model.save_models()


# SAVING MODEL
if False:
    autoencoder.save("autoencoderParams.h5")
    encoder.save("encoderParams.h5")
    decoder.save("decoderParams.h5")



# LOADING MODEL
if False:
    autoencoder2 = tf.keras.models.load_model("autoencoderParams.h5")
    encoder2 = tf.keras.models.load_model("encoderParams.h5")
    decoder2 = tf.keras.models.load_model("decoderParams.h5")
    autoencoder2.compile(optimizer=optimizer, loss=loss)
    encoder2.compile(optimizer=optimizer, loss=loss)
    decoder2.compile(optimizer=optimizer, loss=loss)
    #z_train=encoder2.predict(X_train)
    #z_train = z_train.reshape((-1, np.prod(output_shape_model)))
    #z_test=encoder2.predict(X_test)
    #z_test = z_test.reshape((-1, np.prod(output_shape_model)))
    
    
    
# Generating the TRAIN and TEST features from the trained encoder

ae_feats_train = encoder.predict(X_train)
ae_feats_train_flat = ae_feats_train.reshape((-1, np.prod(output_shape_model)))


ae_feats_test = encoder.predict(X_test)
ae_feats_test_flat = ae_feats_test.reshape((-1, np.prod(output_shape_model)))


print(ae_feats_train_flat.shape)
print(ae_feats_test_flat.shape)











from sklearn.neighbors import NearestNeighbors

classifier = NearestNeighbors(n_neighbors=5, metric="cosine")
classifier.fit(ae_feats_train_flat)



def plot_nearest_images(img_query, imgs_retrieval, outFile):
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(2*n_retrieval, 4))
    fig.suptitle("Results", fontsize=20)

    # Plot query image
    ax = plt.subplot(2, n_retrieval, 0 + 3)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    plt.imshow(img_query)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)  # increase border thickness
        ax.spines[axis].set_color('black')  # set to black
    ax.set_title("Test Image",  fontsize=10)  # set subplot title

    # Plot retrieval images
    for i, img in enumerate(imgs_retrieval):
        ax = plt.subplot(2, n_retrieval, n_retrieval + i + 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
        ax.set_title("NN #%d" % (i+1), fontsize=10)  # set subplot title

    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()


y_pred_5_indices = []
y_pred_1_indices = []

# Save KNN results
for i, emb_flatten in enumerate(ae_feats_test_flat):
    print(i)
    _, indices = classifier.kneighbors([emb_flatten]) # find k nearest train neighbours
    y_pred_5_indices.append([labels_train[i]  for i in (indices[0])])
    y_pred_1_indices.append(labels_train[indices[0][0]])
    print(indices)
    img_query = X_test_raw[i] # query image
    imgs_retrieval = [X_train_raw[idx] for idx in indices.flatten()] # retrieval images
    outFile = os.path.join('out/', "{}_results_{}.png".format('Autoencoder', i))
    #plt.savefig(outFile, bbox_inches='tight')
    
    plot_nearest_images(img_query, imgs_retrieval, outFile)




# ACCURACY
from sklearn.metrics import confusion_matrix, accuracy_score
print('Confusion Matrix on the Nearest Neighbors: ')
print(confusion_matrix(labels_test, y_pred_1_indices, labels=["car", "airplane", "dog", "cat"]))
print(accuracy_score(labels_test, y_pred_1_indices))



print('Confusion Matrix on the aggregate of 5 Nearest Neighbors: ')
y_pred_5_mode = []
for i in range(len(y_pred_5_indices)):
    y_pred_5_mode.append(max(set(y_pred_5_indices[i]), key=y_pred_5_indices[i].count))
    
print(confusion_matrix(labels_test, y_pred_5_mode, labels=["car", "airplane", "dog", "cat"]))
print(accuracy_score(labels_test, y_pred_5_mode))




z1 = list(zip(index_test, labels_test))
z2 = list(zip(index_train, labels_train))
with open('meta_test1.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(z1)
with open('meta_train1.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(z2)
    
    
    
    