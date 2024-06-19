Handwritten Text Recognition using IAM Dataset
Overview
This project involves building an deep learning system to recognize handwritten text using the IAM dataset. The model is built using TensorFlow and Keras, and it employs Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to predict text from images.

Dataset
The IAM dataset is used for training, validation, and testing. This dataset contains images of handwritten words and their corresponding transcriptions. The dataset can be downloaded from the official IAM Handwriting Database.

Project Structure
data/: Contains the IAM dataset images and labels.
IAM_Words.zip: The zipped IAM dataset file.
IAM_Words/: The unzipped content of the IAM dataset.
IAM_Words/words.txt: Text file containing the labels for the images.
IAM_Words/words/: Directory containing the word images.
notebooks/: Jupyter notebooks used for exploratory data analysis and experiments.
saved_model/: Directory to save the trained model.
src/: Source code for the project.

Setup
Prerequisites
Python 3.7+
TensorFlow 2.0+
NumPy
OpenCV
Matplotlib
Installation
Clone the repository:


git clone https://github.com/yourusername/handwriting-recognition.git
cd handwriting-recognition
Install the required packages:



pip install -r requirements.txt
Download and unzip the IAM dataset into the data directory:

wget -q https://git.io/J0fjL -O IAM_Words.zip
unzip -qq IAM_Words.zip -d data
mkdir -p data/words
tar -xf data/IAM_Words/words.tgz -C data/words
mv data/IAM_Words/words.txt data
Preprocessing
The dataset is split into training, validation, and testing sets. Images are resized to 128x32 pixels, and labels are cleaned and vectorized.

Model
The model architecture consists of:

Convolutional Neural Network (CNN) layers for feature extraction.
Recurrent Neural Network (RNN) layers (BiLSTM) for sequence modeling.
A custom CTC (Connectionist Temporal Classification) layer for handling varying lengths of text.
