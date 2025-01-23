#7 Evaluating Caption Results
from pickle import dump, load
from keras.api._tf_keras.keras.applications.vgg16 import VGG16
import numpy as np
from keras.api._tf_keras.keras.models import load_model
from pickle import load
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import rouge

from image_captioning.execute_model import generate_caption, generate_caption_beam
_std_beam_width = 5
_std_vocab_size = 7506
_std_max_length = 33

__is_eval_data_generated = False

def _generate_eval_data(beam_width, vocab_size, max_length):
    # extrahiere trainingsdaten
    fid = open("features.pkl","rb")
    features = load(fid)
    fid.close()

    fid = open("caption_train_tokenizer.pkl","rb")
    caption_train_tokenizer = load(fid)
    fid.close()

    fid = open("image_captions_test.pkl","rb")
    image_captions_test = load(fid)
    fid.close()


    # load the model
    pred_model = load_model('modelConcat_1a_2.h5')

    # wird garnicht verwendet
    # pred_model = load_model('model_3_0.h5')
    # base_model = VGG16(include_top=True) #define the image feature extraction model
    # feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


    #  generiere Image caption kandidaten und speicher diese
    image_captions_candidate = dict()
    for image_fileName, reference_captions in image_captions_test.items():
        image_fileName_feature = image_fileName.split('.')[0]
            
        photo = features[image_fileName_feature]
        image_captions_candidate[image_fileName] = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
        
    fid = open("test_captions_post_concat","wb")
    dump(image_captions_candidate, fid)
    fid.close()

    image_captions_candidate_beam5 = dict()
    for image_fileName, reference_captions in image_captions_test.items():
        image_fileName_feature = image_fileName.split('.')[0]
            
        photo = features[image_fileName_feature]
        image_captions_candidate_beam5[image_fileName], _ = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width)
        #print(image_captions_candidate_beam5[image_fileName])
        
    fid = open("test_captions_concate_beam5_post","wb")
    dump(image_captions_candidate_beam5, fid)
    fid.close()

    __is_eval_data_generated = True


#7.1 BLEU
#greedy bleu
def eval_BLEU(beam_width = _std_beam_width, vocab_size = _std_vocab_size, max_length = _std_max_length):
    if not __is_eval_data_generated:
        _generate_eval_data(beam_width, vocab_size, max_length)
    
    # load data
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



def eval_ROUGE(beam_width = _std_beam_width, vocab_size = _std_vocab_size, max_length = _std_max_length):
    if not __is_eval_data_generated:
        _generate_eval_data(beam_width, vocab_size, max_length)

    # load data
    fid = open("test_captions_post_concat","rb")
    image_captions_candidate = load(fid)
    fid.close()

    fid = open("test_captions_concate_beam5_post","rb")
    image_captions_candidate_beam5 = load(fid)
    fid.close()

    fid = open("image_captions_test.pkl","rb")
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

    #print(rouge_score)

    #rouge beam5
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