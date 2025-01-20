#After the images for each set is identified, we clean up the captions by:
#
#    Remove all numbers and punctuations
#    Change all letters to lower case
#    Remove words containing single characters
#    Add 'START' and 'END' to the target data

# 2.1 Loading Data Sets Image ID
import string
import numpy as np
from pickle import dump
from pickle import load
from collections import Counter
from keras.api._tf_keras.keras.preprocessing.text import Tokenizer
from keras.api._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras.api._tf_keras.keras.utils import to_categorical


def load_data_set_ids(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    dataset = list()
    for image_id in text.split('\n'):
        if len(image_id) < 1:
            continue
            
        dataset.append(image_id)
    
    return set(dataset)

filename = 'Flickr8k.token.txt'
file = open(filename, 'r')
token_text = file.read()
file.close()

translator = str.maketrans("", "", string.punctuation) #translation table that maps all punctuation to None
image_captions = dict()
image_captions_train = dict()
image_captions_dev = dict()
image_captions_test = dict()
image_captions_other = dict()
corpus = list() #corpus used to train tokenizer
corpus.extend(['<START>', '<END>', '<UNK>']) #add SOS and EOS to list first

max_imageCap_len = 0

for line in token_text.split('\n'): # first split on new line
    tokens = line.split(' ') #split on white space, the first segment is 1000268201_693b08cb0e.jpg#0, the following segements are caption texts
    if len(line) < 2:
        continue
    image_id, image_cap = tokens[0], tokens[1:] #use the first segment as image id, the rest as caption
    image_id = image_id.split('#')[0] #strip out #0 from filename
    image_cap = ' '.join(image_cap) #join image caption together again

    image_cap = image_cap.lower() #change to lower case
    image_cap = image_cap.translate(translator) #take out punctuation using a translation table
    
    image_cap = image_cap.split(' ') #split string here because following two methods works on word-level best
    image_cap = [w for w in image_cap if w.isalpha()] #keep only words that are all letters
    image_cap = [w for w in image_cap if len(w)>1]
    image_cap = '<START> ' + ' '.join(image_cap) + ' <END>' #add sentence start/end; note syntax: separator.join()
    
    #update maximum caption length
    if len(image_cap.split()) > max_imageCap_len:
        max_imageCap_len = len(image_cap.split())
    
    #add to dictionary
    if image_id not in image_captions:
        image_captions[image_id] = list() #creat a new list if it does not yet exist
    image_captions[image_id].append(image_cap)
    
    #add to train/dev/test dictionaries
    if image_id in training_set:
        if image_id not in image_captions_train:
            image_captions_train[image_id] = list() #creat a new list if it does not yet exist
        image_captions_train[image_id].append(image_cap)
        corpus.extend(image_cap.split()) #add only training words to corpus to train tokenlizer
        
    elif image_id in dev_set:
        if image_id not in image_captions_dev:
            image_captions_dev[image_id] = list() #creat a new list if it does not yet exist
        image_captions_dev[image_id].append(image_cap)
        
    elif image_id in test_set:
        if image_id not in image_captions_test:
            image_captions_test[image_id] = list() #creat a new list if it does not yet exist
        image_captions_test[image_id].append(image_cap)
    else:
        if image_id not in image_captions_other:
            image_captions_other[image_id] = list() #creat a new list if it does not yet exist
        image_captions_other[image_id].append(image_cap)

caption_train_tokenizer = Tokenizer() #initialize tokenizer
caption_train_tokenizer.fit_on_texts(corpus) #fit tokenizer on training data

# test   
fid = open("image_captions.pkl","wb")
dump(image_captions, fid)
fid.close()

fid = open("image_captions_train.pkl","wb")
dump(image_captions_train, fid)
fid.close()

fid = open("image_captions_dev.pkl","wb")
dump(image_captions_dev, fid)
fid.close()

fid = open("image_captions_test.pkl","wb")
dump(image_captions_test, fid)
fid.close()

fid = open("image_captions_other.pkl","wb")
dump(image_captions_other, fid)
fid.close()

fid = open("caption_train_tokenizer.pkl","wb")
dump(caption_train_tokenizer, fid)
fid.close()

fid = open("corpus.pkl","wb")
dump(corpus, fid)
fid.close()

corpus_count=Counter(corpus)
fid = open("corpus_count.pkl","wb")
dump(corpus_count, fid)
fid.close()

print("size of data =", len(image_captions), "size of training data =", len(image_captions_train), "size of dev data =", len(image_captions_dev), "size of test data =", len(image_captions_test), "size of unused data =", len(image_captions_other))
print("maximum image caption length =",max_imageCap_len)



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
word_index = caption_train_tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, idx in word_index.items():
    embed_vector = embeddings_index.get(word)
    if embed_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[idx] = embed_vector
        
fid = open("embedding_matrix.pkl","wb")
dump(embedding_matrix, fid)
fid.close()



# 2.3 Generating Training Data for Progressive Loading
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
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

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, batch_size, vocab_size):
    # loop for ever over images
    current_batch_size=0
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            if current_batch_size == 0:
                X1, X2, Y = list(), list(), list()
            
            imageFeature_id = key.split('.')[0]
            photo = photos[imageFeature_id][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            #in_img = np.squeeze(in_img)
            X1.extend(in_img)
            X2.extend(in_seq)
            Y.extend(out_word)
            current_batch_size += 1
            if current_batch_size == batch_size:
                current_batch_size = 0
                yield [[np.array(X1), np.array(X2)], np.array(Y)]

# data loader test
fid = open('features.pkl', 'rb')
image_features = load(fid)
fid.close()

# test the data generator
caption_max_length = 33
batch_size = 1
vocab_size = 7057
generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)