# coding: utf-8
"""
Defining global constants
"""
# special tokens
UNK_TOKEN, UNK_ID = "<unk>", 0
PAD_TOKEN, PAD_ID = "<pad>", 1
BOS_TOKEN, BOS_ID = "<s>", 2
EOS_TOKEN, EOS_ID = "</s>", 3

init_vocab = {
    UNK_ID: UNK_TOKEN,
    PAD_ID: PAD_TOKEN,
    BOS_ID: BOS_TOKEN,
    EOS_ID: EOS_TOKEN,
}

special_tokens = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
