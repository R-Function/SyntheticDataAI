from pickle import load
import sys

from train import train
from data_handler import DataHandler
from image_caption_cnn import define_model_concat, get_initial_model

def main() -> int:
    # name für traindata file
    flickr_data_dir     = 'image_captioning/data/Flickr8k/'
    beam_width          = 5
    vocab_size          = 7506
    caption_max_length  = 33
    base_model          = get_initial_model()
    batch_size          = 6
    epochs              = 3

    data_handler = DataHandler(flickr_data_dir)
    print("Data Handler initialized.")
    data_handler.initialize_flicker8k(base_model)
    print("Flicker8k initialized.")
    data_handler.initialize_data()
    print("Data initialized.")
    data_handler.initialize_pretrained_model()
    print("Pretrained Model initialized.")

    fid = open("embedding_matrix.pkl","rb")
    embedding_matrix = load(fid)
    fid.close()
    post_rnn_model_concat = define_model_concat(vocab_size, caption_max_length, embedding_matrix)

    train(model=post_rnn_model_concat, 
          data_handler=data_handler, 
          caption_max_length= caption_max_length,
          vocab_size= vocab_size,
          batch_size=batch_size,
          epochs=epochs)
    


    


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit