import constants

from matplotlib import pyplot as plt
import numpy as np
import string
from pickle import load
from keras.api._tf_keras.keras.models import Model

from data_handler import DataHandler
from image_caption_cnn import define_model_concat

def train(model : Model, 
		  data_handler : DataHandler, 
		  caption_max_length, 
		  vocab_size, 
		  batch_size, 
		  epochs, 
		  destination_dir):
	#4 Training Model
	fid = open(constants.PKL_DATA_FEATURES_PATH,"rb")
	image_features = load(fid)
	fid.close()

	fid = open(constants.PKL_IMG_CAP_TOKENIZER_PATH,"rb")
	caption_train_tokenizer = load(fid)
	fid.close()

	fid = open(constants.PKL_IMG_CAP_TRAIN_PATH,"rb")
	image_captions_train = load(fid)
	fid.close()

	fid = open(constants.PKL_IMG_CAP_DEV_PATH,"rb")
	image_captions_dev = load(fid)
	fid.close()

	steps = len(image_captions_train)
	#steps_per_epoch = np.floor(steps/batch_size)
	post_rnn_model_concat_hist = list()

	for i in range(epochs):
		# create the data generator
		generator = data_handler.data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
		# fit for one epoch
		post_rnn_model_concat_hist.append(model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1,))
		# save model
		model.save(destination_dir+'modelConcat_1_' + str(i) + '.keras')

	history_acc = post_rnn_model_concat_hist[0].history["acc"]
	history_loss = post_rnn_model_concat_hist[0].history["loss"]
	for i in range(epochs-1):
		history_acc = np.concatenate((history_acc, post_rnn_model_concat_hist[i+1]["acc"]), axis=0)
		history_loss = np.concatenate((history_loss, post_rnn_model_concat_hist[i+1]["loss"]), axis=0)

	print(history_acc)
	print(history_loss)

	plt.plot(history_acc)
	plt.savefig(fname=constants.MODEL_HIST_LOSS_PATH)
	plt.plot(history_loss)
	plt.savefit(fname=constants.MODEL_HIST_ACC_PATH)
	