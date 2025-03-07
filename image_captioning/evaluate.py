#7 Evaluating Caption Results
import constants
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pickle import dump, load
from keras.api._tf_keras.keras.applications.vgg16 import VGG16
from keras.api._tf_keras.keras.models import load_model
from pickle import load
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utility.rouge import rouge
from alive_progress import alive_bar;

from execute_model import generate_caption, generate_caption_beam
_std_beam_width = 5
_std_vocab_size = 7506
_std_max_length = 33


def _generate_eval_data(beam_width, vocab_size, max_length, model_path):
    print("\nGenerating Evaluation Data...")
    # extrahiere trainingsdaten
    fid = open(constants.PKL_DATA_FEATURES_PATH,"rb")
    features = load(fid)
    fid.close()

    fid = open(constants.PKL_IMG_CAP_TOKENIZER_PATH,"rb")
    caption_train_tokenizer = load(fid)
    fid.close()

    fid = open(constants.PKL_IMG_CAP_TEST_PATH,"rb")
    image_captions_test = load(fid)
    fid.close()


    print("Loading model...")
    pred_model = load_model(model_path)

 


    #  greedy search for possible caption word candidates
    print(f"Generating most probable caption words for {len(image_captions_test.items())} images.")
    image_captions_candidate = dict()
    with alive_bar(len(image_captions_test.items())) as bar:
        for image_fileName, reference_captions in image_captions_test.items():
            image_fileName_feature = image_fileName.split('.')[0]
                
            photo = features[image_fileName_feature]
            
            try:
                image_captions_candidate[image_fileName] = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
            except UserWarning:
                continue
            bar()
        
    fid = open(constants.EVAL_CAP_DATA_PATH,"wb")
    dump(image_captions_candidate, fid)
    fid.close()

    print(f"\nUsing beam search (width -> {beam_width}) to get best caption for {len(image_captions_test.items())} images.")
    image_captions_candidate_beam5 = dict()
    with alive_bar(len(image_captions_test.items())) as bar:
        for image_fileName, reference_captions in image_captions_test.items():
            image_fileName_feature = image_fileName.split('.')[0]
                
            photo = features[image_fileName_feature]
            image_captions_candidate_beam5[image_fileName], _ = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width)
            bar()
        
    fid = open(constants.EVAL_CAP_BEAM_PATH,"wb")
    dump(image_captions_candidate_beam5, fid)
    fid.close()


def eval_BLEU(model_path,
              beam_width = _std_beam_width, 
              vocab_size = _std_vocab_size, 
              max_length = _std_max_length, 
              is_eval_data_generated = True):
    if not is_eval_data_generated:
        _generate_eval_data(beam_width, vocab_size, max_length, model_path)
    
    # load data
    fid = open(constants.EVAL_CAP_DATA_PATH,"rb")
    image_captions_candidate = load(fid)
    fid.close()

    fid = open(constants.EVAL_CAP_BEAM_PATH,"rb")
    image_captions_candidate_beam5 = load(fid)
    fid.close()

    fid = open(constants.PKL_IMG_CAP_TEST_PATH,"rb")
    image_captions_test = load(fid)
    fid.close()

    chencherry = SmoothingFunction()

    bleu_score = dict()
    bleu_score_beam5 = dict()
    
    print(f"\nEvaluating BLUE score {len(image_captions_test.items())} items.")
    with alive_bar(len(image_captions_test.items())) as bar:
        for image_fileName, reference_captions in image_captions_test.items():
            ref_cap_reformat=list()
            for cap in reference_captions:
                ref_cap_reformat.append(cap.split()[1:-1])
            
            bleu_score[image_fileName] = sentence_bleu(ref_cap_reformat, image_captions_candidate[image_fileName], smoothing_function=chencherry.method1)
            bleu_score_beam5[image_fileName] = sentence_bleu(ref_cap_reformat, list(image_captions_candidate_beam5[image_fileName][-1].split()), smoothing_function=chencherry.method1)
            
            bar()
        
        

    
    bleu_score_array = np.fromiter(bleu_score.values(), dtype=float)
    bleu_score_all =  [str(np.mean(bleu_score_array)),
                       str(np.median(bleu_score_array)),
                       str(np.max(bleu_score_array)),
                       str(np.min(bleu_score_array)),
                       str(np.std(bleu_score_array))]
    bleu_score_beam_5array = np.fromiter(bleu_score_beam5.values(), dtype=float)
    bleu_score_beam_all =  [str(np.mean(bleu_score_beam_5array)),
                            str(np.median(bleu_score_beam_5array)),
                            str(np.max(bleu_score_beam_5array)),
                            str(np.min(bleu_score_beam_5array)),
                            str(np.std(bleu_score_beam_5array))]
    
    # tabelle erzeugen
    row_lable = ["mean", "median", "max", "min", "STD"]
    col_lable = ["BLEU Score", "Greedy", f"Beam Search width {beam_width}"]
    data = np.array([["123456" for _ in range(3)] for _ in range(5)])
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    for i in range(len(row_lable)):
        print(row_lable[i])
        data[i, 0] = row_lable[i]
        data[i, 1] = bleu_score_all[i]
        data[i, 2] = bleu_score_beam_all[i]
    
    table = ax.table(cellText=data,
                    colLabels=col_lable,
                    loc="center")
    table.auto_set_column_width([0, 1, 2])
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(fontsize=10)  # Adjust font size if necessary
        cell.set_edgecolor('black')
    
    plt.savefig(fname=constants.EVAL_BLEU_DEST, pad_inches=0.01)
    plt.show()



def eval_ROUGE(model_path,
               beam_width = _std_beam_width, 
               vocab_size = _std_vocab_size, 
               max_length = _std_max_length, 
               is_eval_data_generated = True):
    if not is_eval_data_generated:
        _generate_eval_data(beam_width, vocab_size, max_length, model_path)

    # load data
    fid = open(constants.EVAL_CAP_DATA_PATH,"rb")
    image_captions_candidate = load(fid)
    fid.close()

    fid = open(constants.EVAL_CAP_BEAM_PATH,"rb")
    image_captions_candidate_beam5 = load(fid)
    fid.close()

    fid = open(constants.PKL_IMG_CAP_TEST_PATH,"rb")
    image_captions_test = load(fid)
    fid.close()
    
    #greedy rouge
    rouge_score = dict()
    for image_fileName, reference_captions in image_captions_test.items():
        cand=[' '.join(image_captions_candidate[image_fileName])]
        ref_cap_reformat=list()
        for cap in reference_captions:
            ref_cap_reformat.append(' '.join(cap.split()[1:-1]))
        
        rouge_score[image_fileName] = rouge(cand, ref_cap_reformat)


    #rouge beam5
    rouge_score_beam5 = dict()
    print(f"\nEvaluating ROUGE score {len(image_captions_test.items())} items.")
    with alive_bar(len(image_captions_test.items())) as bar:
        for image_fileName, reference_captions in image_captions_test.items():
            cand=[image_captions_candidate_beam5[image_fileName][-1]]
            ref_cap_reformat=list()
            for cap in reference_captions:
                ref_cap_reformat.append(' '.join(cap.split()[1:-1]))
            
            rouge_score_beam5[image_fileName] = rouge(cand, ref_cap_reformat)
            bar()


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

    rouge_greedy_mean = [str(np.mean(rouge_1_f_score_array)),
                        str(np.median(rouge_2_f_score_array)),
                        str(np.max(rouge_l_f_score_array))]
    
    rouge_beam_mean = [str(np.mean(rouge_1_f_score_beam5_array)),
                        str(np.median(rouge_2_f_score_beam5_array)),
                        str(np.max(rouge_l_f_score_beam5_array))]

    # tabelle erzeugen
    row_lable = ["rouge_1/f_score", "rouge_2/f_score", "rouge_l/f_score"]
    col_lable = ["Stat", "Greedy Mean", f"Beam Search Mean width {beam_width}"]
    data = np.array([["platzhalterwort" for _ in range(3)] for _ in range(3)])
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    for i in range(len(row_lable)):
        print(row_lable[i])
        data[i, 0] = row_lable[i]
        data[i, 1] = rouge_greedy_mean[i]
        data[i, 2] = rouge_beam_mean[i]
    
    table = ax.table(cellText=data,
                    colLabels=col_lable,
                    loc="center")
    table.auto_set_column_width([0, 1, 2])
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(fontsize=10)
        cell.set_edgecolor('black')
    
    plt.savefig(fname=constants.EVAL_ROUGE_DEST, pad_inches=0.01)
    plt.show()