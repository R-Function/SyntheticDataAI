
# 1. Obtain Features Using Pre-trained Image Models
from pickle import dump
from pickle import load
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from keras.preprocessing.text import Tokenizer
from collections import Counter
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

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


# 2.1 Loading Data Sets Image ID
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

import string
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

#3 Model Definition
from keras.layers.merge import concatenate
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

#4 Training Model
fid = open("features.pkl","rb")
image_features = load(fid)
fid.close()

fid = open("caption_train_tokenizer.pkl","rb")
caption_train_tokenizer = load(fid)
fid.close()

fid = open("image_captions_train.pkl","rb")
image_captions_train = load(fid)
fid.close()

fid = open("image_captions_dev.pkl","rb")
image_captions_dev = load(fid)
fid.close()

caption_max_length = 33
batch_size = 100
vocab_size = 7506
#generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)

#epochs = 2
#steps = len(image_captions_train)
#steps_per_epoch = np.floor(steps/batch_size)

batch_size = 6
steps = len(image_captions_train)
steps_per_epoch = np.floor(steps/batch_size)

epochs = 3

for i in range(epochs):
	# create the data generator
	generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
	# fit for one epoch
	post_rnn_model_concat_hist=post_rnn_model_concat.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	post_rnn_model_concat.save('modelConcat_1_' + str(i) + '.h5')


#5 Creating Captions using Trained Model
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

base_model = VGG16(include_top=True)
feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extract_feature(model, file_name):
    img = load_img(file_name, target_size=(224, 224)) #size is 224,224 by default
    x = img_to_array(img) #change to np array
    x = np.expand_dims(x, axis=0) #expand to include batch dim at the beginning
    x = preprocess_input(x) #make input confirm to VGG16 input format
    fc2_features = model.predict(x)
    return fc2_features

# load the tokenizer
caption_train_tokenizer = load(open('caption_train_tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 33
# load the model
#pred_model = load_model('model_3_0.h5')
pred_model = load_model('modelConcat_1a_2.h5')

def generate_caption(pred_model, caption_train_tokenizer, photo, max_length):
    in_text = '<START>'
    caption_text = list()
    for i in range(max_length):
            # integer encode input sequence
            sequence = caption_train_tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            model_softMax_output = pred_model.predict([photo,sequence], verbose=0)
            # convert probability to integer
            word_index = argmax(model_softMax_output)
            # map integer to word
            word = caption_train_tokenizer.index_word[word_index]
            #print(word)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word != 'end':
                caption_text.append(word)
            if word == 'end':
                break
    return caption_text

caption_image_fileName = 'running-dogs.jpg'
photo = extract_feature(feature_extract_pred_model, caption_image_fileName)
caption = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
print(' '.join(caption))

#6 Beam Search
def flatten(lst):
    return sum( ([x] if not isinstance(x, list) else flatten(x) for x in lst), [] )

def generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width):
    sequence = caption_train_tokenizer.texts_to_sequences(['<START>'])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    model_softMax_output = np.squeeze(pred_model.predict([photo,sequence], verbose=0))
    most_likely_seq=np.argsort(model_softMax_output)[-beam_width:]
    most_likely_prob=np.log(model_softMax_output[most_likely_seq])


    most_likely_cap = list()
    for j in range(beam_width):
        most_likely_cap.append(list())
        most_likely_cap[j] =[[caption_train_tokenizer.index_word[most_likely_seq[j]]]]

    for i in range(max_length):    
        temp_prob = np.zeros((beam_width, vocab_size))
        for j in range(beam_width):
            if most_likely_cap[j][-1] != ['end']: #if not terminated
                num_words = len(most_likely_cap[j])
                sequence = caption_train_tokenizer.texts_to_sequences(most_likely_cap[j])
                sequence = pad_sequences(np.transpose(sequence), maxlen=max_length)
                model_softMax_output = pred_model.predict([photo,sequence], verbose=0)
                temp_prob[j,] = (1/(num_words)) *(most_likely_prob[j]*(num_words-1) + np.log(model_softMax_output)) #update most likily prob
            else:
                temp_prob[j,] = most_likely_prob[j] + np.zeros(vocab_size) - np.inf
                temp_prob[j,0] = most_likely_prob[j]


        x_idx, y_idx = np.unravel_index(temp_prob.flatten().argsort()[-beam_width:], temp_prob.shape)

        most_likely_cap_temp = list()
        for j in range(beam_width):
            most_likely_prob[j] = temp_prob[x_idx[j],y_idx[j]]
            most_likely_cap_temp.append(list())
            most_likely_cap_temp[j] = most_likely_cap[x_idx[j]].copy()
            if most_likely_cap_temp[j][-1] != ['end']:
                most_likely_cap_temp[j].append([caption_train_tokenizer.index_word[y_idx[j]]])

        most_likely_cap = most_likely_cap_temp.copy()

        finished = True
        for j in range(beam_width):
            if most_likely_cap_temp[j][-1] != ['end']:
                finished = False

        if finished == True:
            break

    final_caption = list()

    for j in range(beam_width):
        final_caption.append(' '.join(flatten(most_likely_cap[j][0:-1])))


    return final_caption, most_likely_prob

photo = extract_feature(feature_extract_pred_model, 'running-dogs.jpg')

vocab_size = 7506
beam_width = 10
max_length = 33
caption, prob = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length,vocab_size,beam_width)
print(caption)
print(prob)


#7 Evaluating Caption Results
from pickle import load
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
from numpy import argmax

fid = open("features.pkl","rb")
image_features = load(fid)
fid.close()

fid = open("caption_train_tokenizer.pkl","rb")
caption_train_tokenizer = load(fid)
fid.close()

fid = open("image_captions_test.pkl","rb")
image_captions_test = load(fid)
fid.close()

# pre-define the max sequence length (from training)
max_length = 33
# load the model
pred_model = load_model('modelConcat_1a_2.h5')
#pred_model = load_model('model_3_0.h5')

base_model = VGG16(include_top=True) #define the image feature extraction model
feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

image_captions_candidate = dict()
for image_fileName, reference_captions in image_captions_test.items():
    image_fileName_feature = image_fileName.split('.')[0]
           
    photo = features[image_fileName_feature]
    image_captions_candidate[image_fileName] = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
    
fid = open("test_captions_post_concat","wb")
dump(image_captions_candidate, fid)
fid.close()

fid = open('features.pkl','rb')
features = load(fid)
fid.close()

vocab_size = 7506
beam_width = 5
max_length = 33
final_caption, most_likely_prob = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width)

image_captions_candidate_beam5 = dict()
for image_fileName, reference_captions in image_captions_test.items():
    image_fileName_feature = image_fileName.split('.')[0]
           
    photo = features[image_fileName_feature]
    image_captions_candidate_beam5[image_fileName], _ = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width)
    #print(image_captions_candidate_beam5[image_fileName])
    
fid = open("test_captions_concate_beam5_post","wb")
dump(image_captions_candidate_beam5, fid)
fid.close()


#7.1 BLEU
#greedy bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pickle import load

fid = open("test_captions_post_concat","rb")
image_captions_candidate = load(fid)
fid.close()

fid = open("test_captions_concate_beam5_post","rb")
image_captions_candidate_beam5 = load(fid)
fid.close()

fid = open("image_captions_test.pkl","rb")
image_captions_test = load(fid)
fid.close()

chencherry = SmoothingFunction()

bleu_score = dict()
bleu_score_beam5 = dict()
for image_fileName, reference_captions in image_captions_test.items():
    ref_cap_reformat=list()
    for cap in reference_captions:
        ref_cap_reformat.append(cap.split()[1:-1])
    
    bleu_score[image_fileName] = sentence_bleu(ref_cap_reformat, image_captions_candidate[image_fileName], smoothing_function=chencherry.method1)
    bleu_score_beam5[image_fileName] = sentence_bleu(ref_cap_reformat, list(image_captions_candidate_beam5[image_fileName][-1].split()), smoothing_function=chencherry.method1)
    
    
#print(bleu_score)

import numpy as np
bleu_score_array = np.fromiter(bleu_score.values(), dtype=float)
print('mean bleu='+str(np.mean(bleu_score_array)) + '; median bleu='+str(np.median(bleu_score_array))+'; max bleu='+str(np.max(bleu_score_array))+'; min bleu='+str(np.min(bleu_score_array))+'; std bleu='+str(np.std(bleu_score_array)))

bleu_score_beam_5array = np.fromiter(bleu_score_beam5.values(), dtype=float)
print('mean beam5 bleu='+str(np.mean(bleu_score_beam_5array)) + '; median beam5 bleu='+str(np.median(bleu_score_beam_5array))+'; max beam5 bleu='+str(np.max(bleu_score_beam_5array))+'; min beam5 bleu='+str(np.min(bleu_score_beam_5array))+'; std beam5 bleu='+str(np.std(bleu_score_beam_5array)))


#7.2 ROUGE
#greedy rouge
from rouge import rouge

rouge_score = dict()
for image_fileName, reference_captions in image_captions_test.items():
    cand=[' '.join(image_captions_candidate[image_fileName])]
    ref_cap_reformat=list()
    for cap in reference_captions:
        ref_cap_reformat.append(' '.join(cap.split()[1:-1]))
    
    rouge_score[image_fileName] = rouge(cand, ref_cap_reformat)

#print(rouge_score)

#rouge beam5
from rouge import rouge

rouge_score_beam5 = dict()
for image_fileName, reference_captions in image_captions_test.items():
    cand=[image_captions_candidate_beam5[image_fileName][-1]]
    ref_cap_reformat=list()
    for cap in reference_captions:
        ref_cap_reformat.append(' '.join(cap.split()[1:-1]))
    
    rouge_score_beam5[image_fileName] = rouge(cand, ref_cap_reformat)

#print(rouge_score)

num_test = len(rouge_score_beam5)


rouge_1_f_score_beam5_array = np.zeros(num_test)
rouge_2_f_score_beam5_array = np.zeros(num_test)
rouge_l_f_score_beam5_array = np.zeros(num_test)

idx = 0
for val in rouge_score_beam5.values():
    rouge_1_f_score_beam5_array[idx] = val['rouge_1/f_score']
    rouge_2_f_score_beam5_array[idx] = val['rouge_2/f_score']
    rouge_l_f_score_beam5_array[idx] = val['rouge_l/f_score']
    idx += 1
    

rouge_1_f_score_array = np.zeros(num_test)
rouge_2_f_score_array = np.zeros(num_test)
rouge_l_f_score_array = np.zeros(num_test)

idx = 0
for val in rouge_score.values():
    rouge_1_f_score_array[idx] = val['rouge_1/f_score']
    rouge_2_f_score_array[idx] = val['rouge_2/f_score']
    rouge_l_f_score_array[idx] = val['rouge_l/f_score']
    idx += 1