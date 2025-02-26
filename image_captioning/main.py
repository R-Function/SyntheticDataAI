from pickle import load
import sys

from evaluate import eval_BLEU, eval_ROUGE
from module_test import execute_model_test
from train import train
from data_handler import DataHandler
from image_caption_cnn import define_model_concat, get_initial_model

def main() -> int:
    # name für traindata file
    data_dir            = 'image_captioning/data/Flickr8k/'
    embedd_path         = "image_captioning/data/word_embeddings/glove.6B/glove.6B.50d.txt"
    dest_dir            = "image_captioning/trained_models/"
    beam_width          = 5
    vocab_size          = 7506
    caption_max_length  = 33
    base_model          = get_initial_model()
    batch_size          = 8
    epochs              = 5
    model_path          = 'image_captioning/trained_models/modelConcat_1_2.h5'

    data_handler = DataHandler(data_dir, embedd_path)
    print("\n--> Data Handler initialized.------------------------------------")
    #data_handler.initialize_flicker8k(base_model)
    #print("Flicker8k initialized.")
    data_handler.initialize_data()
    print("\n--> Data initialized.--------------------------------------------")
    data_handler.initialize_pretrained_model()
    print("\n--> Pretrained Model initialized.--------------------------------")

    fid = open("image_captioning/data/Flickr8k/train_data/embedding_matrix.pkl","rb")
    embedding_matrix = load(fid)
    fid.close()
    post_rnn_model_concat = define_model_concat(vocab_size, caption_max_length, embedding_matrix)

    train(model=post_rnn_model_concat, 
          data_handler=data_handler,
          caption_max_length= caption_max_length,
          vocab_size= vocab_size,
          batch_size=batch_size,
          epochs=epochs,
          destination_dir="image_captioning/trained_models/")
    print("--> Model training finished.--------------------------------")
    
    execute_model_test(model_path=model_path, 
                       test_image_path= 'image_captioning/test/beachball_people.jpg')

    # eval_BLEU(model_path = model_path)
    # eval_ROUGE(model_path =model_path)
    


    


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit