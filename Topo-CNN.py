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

#TOPO-CNN MODEL
model_classes = [ResNet50, EfficientNetB0, InceptionV3, MobileNetV2,DenseNet121,ResNet101,VGG16 ] 
epochs_to_test = [5] 
results_list = [] 
for model_class in model_classes:
    for num_epochs in epochs_to_test:
        model_cnn = tf.keras.models.Sequential([model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))])   
        for layer in model_cnn.layers:
            layer.trainable = False
        model_cnn.add(Conv2D(64, (3,3), activation='relu'))
        model_cnn.add(MaxPooling2D(2,2))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(64, activation='relu'))
        def create_MLP(dim, regress=False):
            model = Sequential()
            model.add(Dense(256, input_dim=dim, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(64, activation="relu"))
            return model
        mlp = create_MLP(X_train_tda.shape[1], regress=False)
        combinedInput = concatenate([mlp.output, model_cnn.output])
        x = Dense(256, activation="relu")(combinedInput)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(n_classes, activation='softmax')(x)
        model = Model(inputs=[mlp.input, model_cnn.input], outputs=x)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
        model.fit(x=[X_train_tda, X_train_images], y=y_train,
                  validation_data=([X_test_tda, X_test_images], y_test),
                  epochs=num_epochs, batch_size=64, verbose=0)
        evaluation_results = model.evaluate([X_test_tda, X_test_images], y_test)
        total_loss, accuracy, auc = evaluation_results
        results_list.append({'Model': model_class.__name__, 'Epochs': num_epochs, 'Accuracy': accuracy, 'AUC': auc}) 
results_df = pd.DataFrame(results_list.)
