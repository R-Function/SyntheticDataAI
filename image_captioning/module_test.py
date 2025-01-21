# fremde bibliotheken
import numpy as np
from os import listdir
from pickle import dump, load
from keras.api._tf_keras.keras.models import Model
from keras.api._tf_keras.keras.models import load_model
from keras.api._tf_keras.keras.applications.vgg16 import VGG16
from keras.api._tf_keras.keras.preprocessing.image import load_img
from keras.api._tf_keras.keras.preprocessing.image import img_to_array
from keras.api._tf_keras.keras.applications.vgg16 import preprocess_input
from keras.api._tf_keras.keras.models import Model
# eigene module
from image_captioning.data_loader import DataLoader
from image_captioning.execute_model import extract_feature, generate_caption, generate_caption_beam


def execute_model_test():
    base_model = VGG16(include_top=True)
    feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    caption_train_tokenizer = load(open('caption_train_tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 33
    # load the model
    #pred_model = load_model('model_3_0.h5')
    pred_model = load_model('modelConcat_1a_2.h5')
    
    caption_image_fileName = 'running-dogs.jpg'
    photo = extract_feature(feature_extract_pred_model, caption_image_fileName)
    caption = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
    print(' '.join(caption))

    photo = extract_feature(feature_extract_pred_model, 'running-dogs.jpg')
    vocab_size = 7506
    beam_width = 10
    max_length = 33
    caption, prob = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length,vocab_size,beam_width)
    print(caption)
    print(prob)

#Test erst ausführen, wenn load_initial_data() gemacht wurde
def data_loader_test(data_loader : DataLoader):
    # 2.2 Using Pretrained Embeddings
    embeddings_index = dict()
    fid = open('glove.6B.50d.txt' ,encoding="utf8")
    for line in fid:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    fid.close()

    EMBEDDING_DIM = 50
    word_index = data_loader.caption_train_tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, idx in word_index.items():
        embed_vector = embeddings_index.get(word)
        if embed_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[idx] = embed_vector
            
    fid = open("embedding_matrix.pkl","wb")
    dump(embedding_matrix, fid)
    fid.close()

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


def flicker8k_test(model : Model):
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