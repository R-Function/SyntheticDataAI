# 1. Obtain Features Using Pre-trained Image Models
from pickle import load
from keras.api._tf_keras.keras.utils import plot_model
from keras.api._tf_keras.keras.applications.vgg16 import VGG16
from keras.api._tf_keras.keras.layers import concatenate
from keras.api._tf_keras.keras.layers import Input
from keras.api._tf_keras.keras.layers import Embedding
from keras.api._tf_keras.keras.layers import Dropout
from keras.api._tf_keras.keras.layers import Dense
from keras.api._tf_keras.keras.layers import LSTM
from keras.api._tf_keras.keras.models import Model

def get_initial_model() -> Model:
    base_model = VGG16(include_top=True)
    base_model.summary()

    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    model.summary()
    return model



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
    plot_model(model, to_file='image_captioning/trained_models/model_concat.png', show_shapes=True)
    return model
