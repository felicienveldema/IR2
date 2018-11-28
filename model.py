import torch
import torch.nn as nn
import torch.nn.functional as F

class TopicSeq2Seq(nn.Module):
    def __init__(self, device, input_size, hidden_size, output_size, sos_id, eos_id, max_length, variable_lengths=True):
        super(TopicSeq2Seq, self).__init__()
        self.variable_lengths = variable_lengths
        self.device = device
        self.sos_id = sos_id
        self.encoder = EncoderRNN(device, input_size, hidden_size, variable_lengths)
        self.decoder = DecoderRNN(device, hidden_size, output_size, sos_id, eos_id, max_length)

    def forward(self, input, input_length):
        out, hidden = self.encoder.forward(input.long(), input_length)
        out = self.decoder.forward(self.construct_input_decoder(input.shape[0]), hidden)
        return out

    def construct_input_decoder(self, batch_size):
        input_decoder = []
        for _ in range(batch_size):
            input_decoder.append([self.sos_id])
        return torch.LongTensor(input_decoder).to(self.device)


class EncoderRNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, variable_lengths=True):
        super(EncoderRNN, self).__init__()
        self.variable_lengths = variable_lengths
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, input_length):
        embedded = self.embedding(input)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
        output, hidden = self.gru(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, device, hidden_size, output_size, sos_id, eos_id, max_length):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        sentence = []
        decoder_output = []
        for i in range(self.max_length):
            out, hidden = self.gru(output, hidden)

            out = self.out(out)

            out = self.softmax(out)
            word_id = out.max(2)[1]
            sentence.append(word_id.cpu().numpy()[0][0])
            decoder_output.append(out)
            self.hidden = hidden
            if word_id == self.eos_id:
                break
        return sentence, hidden, decoder_output
