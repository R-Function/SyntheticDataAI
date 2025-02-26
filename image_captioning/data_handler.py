#After the images for each set is identified, we clean up the captions by:
#
#    Remove all numbers and punctuations
#    Change all letters to lower case
#    Remove words containing single characters
#    Add 'START' and 'END' to the target data

# 2.1 Loading Data Sets Image ID
from os import listdir, path
import string
import numpy as np
from pickle import dump
from collections import Counter
from keras.api._tf_keras.keras.preprocessing.text import Tokenizer
from keras.api._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras.api._tf_keras.keras.preprocessing.image import load_img
from keras.api._tf_keras.keras.preprocessing.image import img_to_array
from keras.api._tf_keras.keras.applications.vgg16 import preprocess_input
from keras.api._tf_keras.keras.utils import to_categorical
from keras.api._tf_keras.keras.models import Model

import constants

class DataHandler:
    def __init__(self, 
                 train_data_dir : string,
                 token_path : string,
                 train_set_path : string,
                 dev_set_path : string, 
                 test_set_path : string,
                 embedding_path : string):
        self.train_data_dir = train_data_dir
        self.embedding_path = embedding_path
        file = open(token_path, 'r')
        self.token_text = file.read()
        file.close()

        self.training_set = self._load_data_set_ids(train_set_path)
        self.dev_set = self._load_data_set_ids(dev_set_path)
        self.test_set = self._load_data_set_ids(test_set_path)

        self.translator = str.maketrans("", "", string.punctuation) #translation table that maps all punctuation to None
        self.image_captions             = dict()
        self.image_captions_train       = dict()
        self.image_captions_dev         = dict()
        self.image_captions_test        = dict()
        self.image_captions_other       = dict()
        self.caption_train_tokenizer    = dict()

        self.corpus = list() #corpus used to train tokenizer
        self.corpus.extend(['<START>', '<END>', '<UNK>']) #add SOS and EOS to list first



    def initialize_data(self):
        max_imageCap_len = 0

        for line in self.token_text.split('\n'): # first split on new line
            tokens = line.split(' ') #split on white space, the first segment is 1000268201_693b08cb0e.jpg#0, the following segements are caption texts
            if len(line) < 2:
                continue
            image_id, image_cap = tokens[0], tokens[1:] #use the first segment as image id, the rest as caption
            image_id = image_id.split('#')[0] #strip out #0 from filename
            image_cap = ' '.join(image_cap) #join image caption together again

            image_cap = image_cap.lower() #change to lower case
            image_cap = image_cap.translate(self.translator) #take out punctuation using a translation table
            
            image_cap = image_cap.split(' ') #split string here because following two methods works on word-level best
            image_cap = [w for w in image_cap if w.isalpha()] #keep only words that are all letters
            image_cap = [w for w in image_cap if len(w)>1]
            image_cap = '<START> ' + ' '.join(image_cap) + ' <END>' #add sentence start/end; note syntax: separator.join()
            
            #update maximum caption length
            if len(image_cap.split()) > max_imageCap_len:
                max_imageCap_len = len(image_cap.split())
            
            #add to dictionary
            if image_id not in self.image_captions:
                self.image_captions[image_id] = list() #creat a new list if it does not yet exist
            self.image_captions[image_id].append(image_cap)
            
            #add to train/dev/test dictionaries
            if image_id in self.training_set:
                if image_id not in self.image_captions_train:
                    self.image_captions_train[image_id] = list() #creat a new list if it does not yet exist
                self.image_captions_train[image_id].append(image_cap)
                self.corpus.extend(image_cap.split()) #add only training words to corpus to train tokenlizer
                
            elif image_id in self.dev_set:
                if image_id not in self.image_captions_dev:
                    self.image_captions_dev[image_id] = list() #creat a new list if it does not yet exist
                self.image_captions_dev[image_id].append(image_cap)
                
            elif image_id in self.test_set:
                if image_id not in self.image_captions_test:
                    self.image_captions_test[image_id] = list() #creat a new list if it does not yet exist
                self.image_captions_test[image_id].append(image_cap)
            else:
                if image_id not in self.image_captions_other:
                    self.image_captions_other[image_id] = list() #creat a new list if it does not yet exist
                self.image_captions_other[image_id].append(image_cap)
    
        self.caption_train_tokenizer = Tokenizer() #initialize tokenizer
        self.caption_train_tokenizer.fit_on_texts(self.corpus) #fit tokenizer on training data

        # test   
        fid = open(constants.PKL_IMG_CAP_PATH,"wb")
        dump(self.image_captions, fid)
        fid.close()

        fid = open(constants.PKL_IMG_CAP_TRAIN_PATH,"wb")
        dump(self.image_captions_train, fid)
        fid.close()

        fid = open(constants.PKL_IMG_CAP_DEV_PATH,"wb")
        dump(self.image_captions_dev, fid)
        fid.close()

        fid = open(constants.PKL_IMG_CAP_TEST_PATH,"wb")
        dump(self.image_captions_test, fid)
        fid.close()

        fid = open(constants.PKL_IMG_CAP_OTHER_PATH,"wb")
        dump(self.image_captions_other, fid)
        fid.close()

        fid = open(constants.PKL_IMG_CAP_TOKENIZER_PATH,"wb")
        dump(self.caption_train_tokenizer, fid)
        fid.close()

        fid = open(constants.PKL_IMG_CAP_CORPUS_PATH,"wb")
        dump(self.corpus, fid)
        fid.close()

        corpus_count=Counter(self.corpus)
        fid = open(constants.PKL_IMG_CAP_CORP_COUNT_PATH,"wb")
        dump(corpus_count, fid)
        fid.close()

        print("size of data =", len(self.image_captions), "size of training data =", len(self.image_captions_train), "size of dev data =", len(self.image_captions_dev), "size of test data =", len(self.image_captions_test), "size of unused data =", len(self.image_captions_other))
        print("maximum image caption length =",max_imageCap_len)


    def initialize_pretrained_model(self):
        EMBEDDING_DIM   = 50
        glove_embedding = path.join(self.embedding_path)

        embeddings_index = dict()
        fid = open(glove_embedding,encoding="utf8")
        for line in fid:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        fid.close()


        word_index = self.caption_train_tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

        for word, idx in word_index.items():
            embed_vector = embeddings_index.get(word)
            if embed_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[idx] = embed_vector
                
        fid = open(constants.PKL_EMBED_MATRIX_PATH,"wb")
        dump(embedding_matrix, fid)
        fid.close()

    def initialize_flicker8k(self, model : Model):
        features = dict()
        for file in listdir(self.train_data_dir):
            img_path = path.join(self. train_data_dir, file)
            img = load_img(img_path, target_size=(224, 224)) #size is 224,224 by default
            x = img_to_array(img) #change to np array
            x = np.expand_dims(x, axis=0) #expand to include batch dim at the beginning
            x = preprocess_input(x) #make input confirm to VGG16 input format
            fc2_features = model.predict(x)
            
            name_id = file.split('.')[0] #take the file name and use as id in dict
            features[name_id] = fc2_features

        dump(features, open(constants.PKL_DATA_FEATURES_PATH, 'wb')) #cannot use JSON because ndarray is not JSON serializable


    # data generator, intended to be used in a call to model.fit_generator()
    def data_generator(self, descriptions, photos, tokenizer, max_length, batch_size, vocab_size):
        # loop for ever over images
        current_batch_size=0
        while 1:
            for key, desc_list in descriptions.items():
                # retrieve the photo feature
                if current_batch_size == 0:
                    X1, X2, Y = list(), list(), list()
                
                imageFeature_id = key.split('.')[0]
                photo = photos[imageFeature_id][0]
                in_img, in_seq, out_word = self._create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                #in_img = np.squeeze(in_img)
                X1.extend(in_img)
                X2.extend(in_seq)
                Y.extend(out_word)
                current_batch_size += 1
                if current_batch_size == batch_size:
                    current_batch_size = 0
                    #print("Shape of X1", np.array(X1).shape)
                    #print("Shape of X2", np.array(X2).shape)
                    #print("Shape of Y", np.array(Y).shape)
                    yield (np.array(X1), np.array(X2)), np.array(Y)


    # 2.3 Generating Training Data for Progressive Loading
    def _create_sequences(self, tokenizer, max_length, desc_list, photo, vocab_size):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0] #[0] is used to take out the extra dim. This changes from text to a number
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                # import pdb; pdb.set_trace()
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(np.squeeze(X1)), np.array(X2), np.array(y)

    def _load_data_set_ids(self, filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        
        dataset = list()
        for image_id in text.split('\n'):
            if len(image_id) < 1:
                continue   
            dataset.append(image_id)

        return set(dataset)