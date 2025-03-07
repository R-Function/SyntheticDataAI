import sys
import argparse
import json
import constants

from pickle import load
from evaluate import eval_BLEU, eval_ROUGE
from module_test import execute_model_test
from train import train
from data_handler import DataHandler
from image_caption_cnn import define_model_concat, get_initial_model

def main() -> int:
    EXECECUTION_CHOICES = {"prep-data"      : prep_data, 
                           "train"          : train_model, 
                           "execute-model"  : execute_model, 
                           "evaluate"       : evaluate}
    
    # Command line Arguments
    parser = argparse.ArgumentParser("Interface for data preperation, training, executing and evaluating the Image Captioning RNN.")
    parser.add_argument("config_file", 
                        help= "The path of the config file. Needs to be a .json document.")
    parser.add_argument("-e",
                        "--execute", 
                        help= "Specify what part of the programm you want to run. Note that the data only has to be prepped once and then after it has been changed. If this option is left open, everything will be run in order.",
                        choices=EXECECUTION_CHOICES.keys())
    args = parser.parse_args()

    # JSON Parser
    with open(args.config_file) as fp:# hier argsparse argument rein
        config = json.load(fp)

    # DataHandler Object
    data_handler = DataHandler(train_data_dir   = config["data files and directories"]["data_dir"],
                               token_path       = config["data files and directories"]["token_path"],
                               train_set_path   = config["data files and directories"]["train_set_path"],
                               dev_set_path     = config["data files and directories"]["dev_set_path"],
                               test_set_path    = config["data files and directories"]["test_set_path"],
                               embedding_path   = config["data files and directories"]["embedd_path"])
    print("\t--> Data Handler initialized!")
    
    if args.execute == None:
        prep_data(config = config, data_handler = data_handler)
        train_model(config = config, data_handler = data_handler)
        execute_model(config = config, data_handler = data_handler)
        evaluate(config = config, data_handler = data_handler)
    else:
        EXECECUTION_CHOICES[args.execute](config = config, data_handler = data_handler)


def prep_data(config, data_handler):
    data_handler.extract_features(get_initial_model())
    print("\t--> Features Extracted.")

    data_handler.initialize_data()
    print("\t--> Data initialized.")

    data_handler.initialize_pretrained_model()
    print("\t--> Pretrained Model initialized.")

def train_model(config, data_handler):
    with open(constants.PKL_EMBED_MATRIX_PATH,"rb") as fid:
        embedding_matrix = load(fid)

    post_rnn_model_concat = define_model_concat(vocab_size       = config["hyperparams"]["vocab_size"],
                                                max_length       = config["hyperparams"]["caption_max_length"], 
                                                embedding_matrix = embedding_matrix,
                                                embedding_dim    = config["hyperparams"]["embedding_dim"])

    train(model              = post_rnn_model_concat, 
          data_handler       = data_handler,
          caption_max_length = config["hyperparams"]["caption_max_length"],
          vocab_size         = config["hyperparams"]["vocab_size"],
          batch_size         = config["hyperparams"]["batch_size"],
          epochs             = config["hyperparams"]["epochs"],
          destination_dir    = config["data files and directories"]["dest_dir"])
    
    print("\t--> Model training finished!")

def execute_model(config, data_handler):
    execute_model_test(model_path       = config["data files and directories"]["model_path"], 
                       test_image_path  = config["data files and directories"]["test_img_path"],
                       vocab_size       = config["hyperparams"]["vocab_size"],
                       beam_width       = config["hyperparams"]["beam_search_width"],
                       max_length       = config["hyperparams"]["caption_max_length"])
    
def evaluate(config, data_handler):
    eval_BLEU(model_path  = config["data files and directories"]["model_path"])
    eval_ROUGE(model_path = config["data files and directories"]["model_path"])

if __name__ == '__main__':
    sys.exit(main())