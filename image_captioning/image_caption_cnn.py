# 1. Obtain Features Using Pre-trained Image Models
from pickle import dump
from pickle import load
from os import listdir
from keras.api._tf_keras.keras.utils import plot_model
from keras.api._tf_keras.keras.applications.vgg16 import VGG16
from keras.api._tf_keras.keras.preprocessing.image import load_img
from keras.api._tf_keras.keras.preprocessing.image import img_to_array
from keras.api._tf_keras.keras.applications.vgg16 import preprocess_input
from keras.api._tf_keras.keras.layers.merge import concatenate
from keras.api._tf_keras.keras.layers import Input
from keras.api._tf_keras.keras.layers import Embedding
from keras.api._tf_keras.keras.layers import Dropout
from keras.api._tf_keras.keras.layers import Dense
from keras.api._tf_keras.keras.models import Model
import numpy as np
from collections import Counter

base_model = VGG16(include_top=True)
base_model.summary()

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model.summary()

features = dict()
for file in listdir('Flicker8k_Dataset'):
    img_path = 'Flicker8k_Dataset/' + file
    img = load_img(img_path, target_size=(224, 224)) #size is 224,224 by default
    x = img_to_array(img) #change to np array
    x = np.expand_dims(x, axis=0) #expand to include batch dim at the beginning
    x = preprocess_input(x) #make input confirm to VGG16 input format
    fc2_features = model.predict(x)
    
    name_id = file.split('.')[0] #take the file name and use as id in dict
    features[name_id] = fc2_features

dump(features, open('features.pkl', 'wb')) #cannot use JSON because ndarray is not JSON serializable

#3 Model Definition
def define_model_concat(vocab_size, max_length, embedding_matrix):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    image_feature = Dropout(0.5)(inputs1)
    image_feature = Dense(256, activation='relu')(image_feature)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
    #Embedding(vocab_size, 256, mask_zero=True)(inputs2) #<<<<<< fix me, add pretrianed embedding
    language_feature = Dropout(0.5)(language_feature)
    language_feature = LSTM(256)(language_feature)
    # decoder model
    output = concatenate([image_feature, language_feature])
    output = Dense(256, activation='relu')(output)
    output = Dense(vocab_size, activation='softmax')(output)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model_concat.png', show_shapes=True)
    return model

fid = open("embedding_matrix.pkl","rb")
embedding_matrix = load(fid)
fid.close()

caption_max_length = 33
vocab_size = 7506
post_rnn_model_concat = define_model_concat(vocab_size, caption_max_length, embedding_matrix)