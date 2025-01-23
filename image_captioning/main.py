from pickle import load
import sys

from image_captioning.train import train
from image_captioning.data_handler import DataHandler
from image_captioning.image_caption_cnn import define_model_concat, get_initial_model

def main() -> int:
    # name für traindata file
    filename            = 'image_captioning/data/Flickr8k/Flickr8k_text/Flickr8k.token.txt'
    beam_width          = 5
    vocab_size          = 7506
    caption_max_length  = 33
    base_model          = get_initial_model()
    batch_size          = 6
    epochs              = 3

    data_loader = DataHandler(filename)
    data_loader.initialize_flicker8k(base_model)
    data_loader.initialize_data()
    data_loader.initialize_pretrained_model()

    fid = open("embedding_matrix.pkl","rb")
    embedding_matrix = load(fid)
    fid.close()
    post_rnn_model_concat = define_model_concat(vocab_size, caption_max_length, embedding_matrix)

    train(model=post_rnn_model_concat, 
          data_loader=data_loader, 
          caption_max_length= caption_max_length,
          vocab_size= vocab_size,
          batch_size=batch_size,
          epochs=epochs)
    


    


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit