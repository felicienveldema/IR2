from model import TopicSeq2Seq
import build_vocab
import data_loader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np

DNN_HIDDEN_UNITS_DEFAULT = 128
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 1
EVAL_FREQ_DEFAULT = 100
SRC_PATH = "Data/input_data.txt"
TRG_PATH = "Data/target_data.txt"
SRC_WORD2ID_PATH = "Data/input_word2id.json"
TRG_WORD2ID_PATH = "Data/target_word2id.json"
DEVICE = "cuda:0"
SOS_ID = 1
EOS_ID = 2
MAX_LENGTH = 5
def retrieve_data():
    if not os.path.isfile(SRC_WORD2ID_PATH):
        build_vocab.main(SRC_PATH, TRG_PATH, SRC_WORD2ID_PATH, TRG_WORD2ID_PATH)

    with open(SRC_WORD2ID_PATH, 'r') as f:
        src_word2id = json.load(f)
    with open(TRG_WORD2ID_PATH, 'r') as f:
        trg_word2id = json.load(f)
    vocab_size_src = len(src_word2id)
    vocab_size_trg = len(trg_word2id)
    dataloader = data_loader.get_loader(src_path=SRC_PATH,
                             trg_path=TRG_PATH,
                             src_word2id=src_word2id,
                             trg_word2id=trg_word2id,
                             batch_size=FLAGS.batch_size)


    return dataloader, vocab_size_src, vocab_size_trg

def one_hot(input_data, target_length, vocab_size):
    one_hot = np.zeros((target_length, vocab_size))
    one_hot[np.arange(target_length), input_data] = 1
    return torch.LongTensor(one_hot).view(1,target_length,-1).to(DEVICE)

def main():
    dataloader, vocab_size_src, vocab_size_trg = retrieve_data()
    topic_seq2seq = TopicSeq2Seq(FLAGS.device, vocab_size_src, FLAGS.dnn_hidden_units, vocab_size_trg, SOS_ID, EOS_ID, FLAGS.max_length).to(FLAGS.device)
    encoder_optimizer = optim.Adam(topic_seq2seq.encoder.parameters(), lr=FLAGS.learning_rate)
    decoder_optimizer = optim.Adam(topic_seq2seq.decoder.parameters(), lr=FLAGS.learning_rate)
    for step, (batch_input, input_length, batch_target, target_length) in enumerate(dataloader):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        out = topic_seq2seq.forward(batch_input.to(FLAGS.device), input_length)
        print(out[2])
        one_hot_targets = one_hot(batch_target, target_length[0], vocab_size_trg)
        one_hot_out = one_hot(out[0], len(out[0]), vocab_size_trg)

        loss = nn.NLLLoss()(out[0], one_hot_targets)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        break


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    # parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
    #                     help='Directory for storing input data')
    parser.add_argument('--device', type=str, default=DEVICE, help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--max_length', type = int, default = MAX_LENGTH,
                      help='Max length of generated sentence')
    FLAGS, unparsed = parser.parse_known_args()

    main()