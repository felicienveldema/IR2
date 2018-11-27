import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda:0"

class TopicSeq2Seq(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, variable_lengths=True):
		super(TopicSeq2Seq, self).__init__()
		self.variable_lengths = variable_lengths
		self.encoder = EncoderRNN(input_size, hidden_size, variable_lengths)
		self.decoder = DecoderRNN(hidden_size, output_size)

	def forward(self, input, input_length):
		out, hidden = self.encoder.forward(input, input_length)
		out = self.decoder.forward(out, hidden)
		return out


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, variable_lengths=True):
        super(EncoderRNN, self).__init__()
        self.variable_lengths = variable_lengths
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, input_length):
        embedded = self.embedding(input)
        if self.variable_lengths:
        	embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
        output, hidden = self.gru(embedded)
        if self.variable_lengths:
        	output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, input_length, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        self.hidden = hidden
        return output, hidden