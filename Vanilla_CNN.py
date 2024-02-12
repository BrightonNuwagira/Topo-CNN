from tqdm import tqdm 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as data 
import torchvision.transforms as transforms 
import medmnist 
from medmnist import INFO, Evaluator 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 
from tensorflow.keras.applications.resnet import ResNet50 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNetV2 
from tensorflow.keras.applications.densenet import DenseNet121 
from tensorflow.keras.applications import ResNet101 
import tensorflow as tf 
from tensorflow.keras.layers import Input 
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D 
from skimage.transform import resize 
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt 
from keras import Model 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, concatenate 
from tensorflow.keras import Input 


for X_train_images, y_train in train_loader:
    print(X_train_images.shape,y_train.shape )
    break 
X_train_images = X_train_images.numpy().transpose(0, 2, 3, 1) 
X_train_images = np.array([resize(image, (224, 224,3)) for image in X_train_images]) 

for X_test_images,y_test in test_loader:
    print(X_test_images.shape,y_test.shape )
    break 
X_test_images = X_test_images.numpy().transpose(0, 2, 3, 1) 
X_test_images = np.array([resize(image, (224, 224)) for image in X_test_images]) 


df_betti_test= pd.read_csv('Test_Dermamnist_Betti.csv') 
X_test_tda=df_betti_test.iloc[:, :-1] 
df_betti_train= pd.read_csv('Train_Dermamnist_Betti.csv') 
X_train_tda=df_betti_train.iloc[:, :-1] 

n_classes = 7
y_train=df_betti_train.iloc[:, -1] 
label_encoder = LabelEncoder() 
y_train = label_encoder.fit_transform(y_train) 
y_train = tf.keras.utils.to_categorical(y_train, n_classes) 

y_test=df_betti_test.iloc[:, -1] 
y_test = label_encoder.fit_transform(y_test) 
y_test = tf.keras.utils.to_categorical(y_test, n_classes) 

seed_value = 42 
np.random.seed(seed_value) 
tf.random.set_seed(seed_value)


#VANILLA-CNN MODEL
epochs_to_test = [5] 
models_to_test = [ResNet50, EfficientNetB0, InceptionV3, MobileNetV2,DenseNet121,ResNet101,VGG16 ] 
results_list = [] 

for model_class in models_to_test:
    model_name = model_class.__name__
    
    for num_epochs in epochs_to_test:
        print(f"Training {model_name} for {num_epochs} epochs...")
        base_model = model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
        model.fit(x=[X_train_images], y=y_train, validation_data=([X_test_images], y_test),
                  epochs=num_epochs, batch_size=64, verbose=0)
        evaluation_results = model.evaluate(X_test_images, y_test)
        total_loss, accuracy, auc = evaluation_results
        results_list.append({'Model': model_name, 'Epochs': num_epochs, 'Accuracy': accuracy, 'AUC': auc}) 
results_df = pd.DataFrame(results_list) 



