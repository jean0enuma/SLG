from transformers import AutoTokenizer
DATASET="phoenix"
MAX_LENGTH=230
MIN_LENGTH=32
PRETRAIN_MAX_LENGTH=200
PRETRAIN_MIN_LENGTH=64
TEMP_SCALING=0.2
#MODEL_PATH="/media/jean/data_storage/CSLR/Models/german_bert_preprocessed_deberta"
MODEL_PATH="/media/jean/data_storage/CSLR/Models/german_bert_preprocessed_deberta"
#TOKENIZER=AutoTo kenizer.from_pretrained(MODEL_PATH)