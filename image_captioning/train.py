import numpy as np
import string
from pickle import load
from keras.api._tf_keras.keras.models import Model

from data_handler import DataHandler
from image_caption_cnn import define_model_concat

def train(model : Model, data_handler : DataHandler, flickr_train_data_dir : string, caption_max_length, vocab_size, batch_size, epochs):
	

	#4 Training Model
	fid = open(flickr_train_data_dir+"features.pkl","rb")
	image_features = load(fid)
	fid.close()

	fid = open(flickr_train_data_dir+"caption_train_tokenizer.pkl","rb")
	caption_train_tokenizer = load(fid)
	fid.close()

	fid = open(flickr_train_data_dir+"image_captions_train.pkl","rb")
	image_captions_train = load(fid)
	fid.close()

	fid = open(flickr_train_data_dir+"image_captions_dev.pkl","rb")
	image_captions_dev = load(fid)
	fid.close()

	#generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)

	#epochs = 2
	#steps = len(image_captions_train)
	#steps_per_epoch = np.floor(steps/batch_size)

	steps = len(image_captions_train)
	steps_per_epoch = np.floor(steps/batch_size)

	epochs = 3

	for i in range(epochs):
		# create the data generator
		generator = data_handler.data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
		# fit for one epoch
		post_rnn_model_concat_hist=model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
		# save model
		model.save('image_captioning/trained_models/modelConcat_1_' + str(i) + '.keras')
