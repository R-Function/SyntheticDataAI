import numpy as np
from pickle import load
from keras.api._tf_keras.keras.models import Model

from image_captioning.data_handler import DataHandler
from image_captioning.image_caption_cnn import define_model_concat

def train(post_rnn_model_concat : Model, data_loader : DataHandler, caption_max_length, vocab_size, batch_size, epochs):
	

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

	#generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)

	#epochs = 2
	#steps = len(image_captions_train)
	#steps_per_epoch = np.floor(steps/batch_size)

	steps = len(image_captions_train)
	steps_per_epoch = np.floor(steps/batch_size)

	epochs = 3

	for i in range(epochs):
		# create the data generator
		generator = data_loader.data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
		# fit for one epoch
		post_rnn_model_concat_hist=post_rnn_model_concat.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
		# save model
		post_rnn_model_concat.save('modelConcat_1_' + str(i) + '.h5')
