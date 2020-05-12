# Image Captioning with Autoencoders
In this project we aim to caption an image using a combination of autoencoders and SVM. For this purpose we use a subset of the captioned MS-COCO dataset for training. This is achieved in three board stages:

## A. Feature Extraction with Autoencoder
In this stage we use a Convolutional Autoencoder to compress the images into a smaller feature space.
The autoencoder minimizes the original image (200px x 200px RGB) into a smaller feature space.
It also minimizes the loss by reconstucting an image from the smaller feature space and applying gradient descent to readjust weights.

## B. Label Prediction with SVM trained on Autoencoder Features
Once the autoencoder has been trained, the new features are extracted for the training and test set.
An SVM model is then trained with the training data features and label predicions are made to determine the category of the test images.

## C. Caption Prediction using the Nearest Neighbors Algorithm
Lastly, the nearest neighbors algorithms is used to find the K=5 nearest images within the predicted class.
The most appropriate caption is determined from the captions of the 5 nearest neighbors identified within the class. Its done using semantic similarity analysis among the captions.

## Preview Notebook
- AutoCaption.ipynb can we previewed but it will not contain all the visualizations (specifically the altair plots).
- Use AutoCaption_Rendered.html to preview all the visualizations.

## Environment Setup:

0. Download the codebase.

1. Download the Fast Text 300D embeddings, unzip it and place the wiki-news-300d-1M.vec file in the 'models' directory.
Link: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

2. Install all the python dependencies requirements using the requirements.txt

3. Open the AutoCaption.ipynb notebook using Jupyter.

4. Make sure that the 'download_coco' flag is set to True in the notebook and then execute it.

## Example Results
### 1. Caption Recommendation

![alt text](https://raw.githubusercontent.com/ankushjain2001/aj2885_das968_Autoencoder_Image_Captioning/master/output/m_100eps_mse_adam_aug6739_32_16_16_5x5/000000024020_results.png)

### 2. Autoencode Feature Reconstruction

![alt text](https://raw.githubusercontent.com/ankushjain2001/aj2885_das968_Autoencoder_Image_Captioning/master/output/m_100eps_mse_adam_aug6739_32_16_16_5x5/reconstructed_8.png)
