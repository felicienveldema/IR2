import nltk
import json
import argparse
from collections import Counter


def build_word2id(seq_path, min_word_count):
    """Creates word2id dictionary.
    
    Args:
        seq_path: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    sequences = open(seq_path).readlines()
    num_seqs = len(sequences)
    counter = Counter()
    
    for i, sequence in enumerate(sequences):
        tokens = nltk.tokenize.word_tokenize(sequence.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<pad>'] = 0
    word2id['<start>'] = 1
    word2id['<end>'] = 2
    word2id['<unk>'] = 3
    
    # if word frequency is less than 'min_word_count', then the word is discarded
    words = [word for word, count in counter.items() if count >= min_word_count]
    
    # add the words to the word2id dictionary
    for i, word in enumerate(words):
        word2id[word] = i + 4
    
    return word2id


def main(src_path, trg_path, src_word2id_path, trg_word2id_path, min_word_count=4):
    
    # build word2id dictionaries for source and target sequences

    src_word2id = build_word2id(src_path, min_word_count)
    trg_word2id = build_word2id(trg_path, min_word_count)
    
    # save word2id dictionaries
    with open(src_word2id_path, 'w') as f:
        json.dump(src_word2id, f)
    with open(trg_word2id_path, 'w') as f:
        json.dump(trg_word2id, f)