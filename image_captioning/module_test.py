# fremde bibliotheken
import numpy as np
from os import listdir
from pickle import load
from keras.api._tf_keras.keras.models import Model
from keras.api._tf_keras.keras.models import load_model
from keras.api._tf_keras.keras.applications.vgg16 import VGG16
from keras.api._tf_keras.keras.models import Model
# eigene module
from data_handler import DataHandler
from execute_model import extract_feature, generate_caption, generate_caption_beam


def execute_model_test():
    base_model = VGG16(include_top=True)
    feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    caption_train_tokenizer = load(open('image_captioning/data/Flickr8k/train_data/caption_train_tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 33
    # load the model
    #pred_model = load_model('model_3_0.h5')
    pred_model = load_model('image_captioning/trained_models/modelConcat_1_2.h5')
    
    caption_image_fileName = 'image_captioning/test/beachball_people.jpg'
    photo = extract_feature(feature_extract_pred_model, caption_image_fileName)
    caption = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
    print(' '.join(caption))

    photo = extract_feature(feature_extract_pred_model, caption_image_fileName)
    vocab_size = 7506
    beam_width = 10
    max_length = 33
    caption, prob = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length,vocab_size,beam_width)
    print(caption)
    print(prob)

#Test erst ausführen, wenn load_initial_data() gemacht wurde
def data_loader_test(data_loader : DataHandler):
    

    fid = open('features.pkl', 'rb')
    image_features = load(fid)
    fid.close()

    caption_max_length = 33
    batch_size = 1
    vocab_size = 7057
    generator = data_loader.data_generator(data_loader.image_captions_train, image_features, data_loader.caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    inputs, outputs = next(generator)
    print(inputs[0].shape)
    print(inputs[1].shape)
    print(outputs.shape)


