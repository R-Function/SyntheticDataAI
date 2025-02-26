from os import path
# filepaths for preprocessed data
PKL_DATA_DIR = "image_captioning\data\Flickr8k"

PKL_IMG_CAP_PATH            = path.join(PKL_DATA_DIR,"image_captions.pkl")
PKL_IMG_CAP_TRAIN_PATH      = path.join(PKL_DATA_DIR,"image_captions_train.pkl")
PKL_IMG_CAP_DEV_PATH        = path.join(PKL_DATA_DIR,"image_captions_dev.pkl")
PKL_IMG_CAP_TEST_PATH       = path.join(PKL_DATA_DIR,"image_captions_test.pkl")
PKL_IMG_CAP_OTHER_PATH      = path.join(PKL_DATA_DIR,"image_captions_other.pkl")
PKL_IMG_CAP_TOKENIZER_PATH  = path.join(PKL_DATA_DIR,"caption_train_tokenizer.pkl")
PKL_IMG_CAP_CORPUS_PATH     = path.join(PKL_DATA_DIR,"corpus.pkl")
PKL_IMG_CAP_CORP_COUNT_PATH = path.join(PKL_DATA_DIR,"corpus_count.pkl")
PKL_DATA_FEATURES_PATH      = path.join(PKL_DATA_DIR,'features.pkl')
PKL_EMBED_MATRIX_PATH       = path.join(PKL_DATA_DIR,'embedding_matrix.pkl')

# filepaths for data to be evaluated
EVAL_DATA_DIR = "image_captioning/data/eval_data/"

EVAL_CAP_DATA_PATH  = path.join(EVAL_DATA_DIR,"test_captions_post_concat")
EVAL_CAP_BEAM_PATH  = path.join(EVAL_DATA_DIR,"test_captions_concate_beam5_post")
EVAL_BLEU_DEST      = path.join(EVAL_DATA_DIR,"BLEU_Score.png")
EVAL_ROUGE_DEST     = path.join(EVAL_DATA_DIR,"ROUGE_Score.png")

# filepath for image captioning model diagramm
MODEL_DIAGRAMM_PATH  = 'image_captioning/trained_models/model_concat.png'
