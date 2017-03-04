import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        inputSize = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

        # self.rnn.bias_ih_l0.data.div_(2)
        # self.rnn.bias_hh_l0.data.copy_(self.rnn.bias_ih_l0.data)

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

    def forward(self, input, hidden=None):
        batch_size = input.size(0) # batch first for multi-gpu compatibility
        emb = self.word_lut(input).transpose(0, 1)
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)

        outputs, hidden_t = self.rnn(emb, hidden)
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            layer = nn.LSTMCell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_%d' % i)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        # self.rnn.bias_ih.data.div_(2)
        # self.rnn.bias_hh.data.copy_(self.rnn.bias_ih.data)

        self.hidden_size = opt.rnn_size

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)


    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input).transpose(0, 1)

        batch_size = input.size(0)

        h_size = (batch_size, self.hidden_size)
        output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for i, emb_t in enumerate(emb.chunk(emb.size(0), dim=0)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs.transpose(0, 1), hidden, attn


class TempEstimator(nn.Module):
    def __init__(self, opt):
        super(TempEstimator, self).__init__()
        if opt.brnn:
            self.linear1 = nn.Linear(opt.rnn_size*opt.layers*opt.batch_size*2, 1)
        else:
            self.linear1 = nn.Linear(opt.rnn_size * opt.layers * opt.batch_size, 1)
        self.softplus = nn.Softplus()
        # self.relu = nn.ReLU()

    def forward(self, input):
        out = self.linear1(input)
        temp = self.softplus(out)
        # temp = self.relu(out)

        return temp

class GANGenerator(nn.Module):

    def __init__(self, opt, dicts):
        self.opt = opt
        self.dicts = dicts
        self.iter_cnt = 0 # retains the overall number of trin iterations

        super(GANGenerator, self).__init__()
        self.eps = 1e-20
        self.real_temp = 0.5
        if not self.opt.estimate_temp:
            self.tau0 = 0.5  # initial temperature
            self.scheduled_temp = self.tau0
            self.ANNEAL_RATE = 0.00003
            self.MIN_TEMP = 0.5
        else:
            self.temp_estimator = TempEstimator(self.opt)
            self.learned_temp = 0
        self.linear = nn.Linear(opt.rnn_size, self.dicts.size())


    def anneal_tau_temp(self):
        # Anneal temperature tau
        self.scheduled_temp = np.maximum(self.tau0 *
                                      np.exp(-self.ANNEAL_RATE * self.iter_cnt * self.opt.batch_size),
                                      self.MIN_TEMP)
        print('Temperature annealed to: ' + str(self.scheduled_temp))

    def get_noise(self, input):
        noise = torch.rand(input.size())
        if self.opt.cuda:
            noise = noise.cuda()
        noise.add_(self.eps).log_().neg_()
        noise.add_(self.eps).log_().neg_()
        noise = Variable(noise)
        return noise

    def real_sampler(self, input):
        noise = self.get_noise(input)
        x = (input + noise)
        # x = x / self.real_temp
        x = x / self.learned_temp
        # x = x * self.real_temp
        return x.view_as(input)

    def sampler(self, input, temp_estim=None):
        noise = self.get_noise(input)
        x = (input + noise)

        if temp_estim:
            # x = x * temp_estim.repeat(x.size())
            x = x / temp_estim.repeat(x.size())
        else:
            # x = x * self.scheduled_temp
            x = x / self.scheduled_temp
        return x.view_as(input)

    def forward(self, input, hidden=None):
        out = self.linear(input)
        if self.opt.use_gumbel:
            temp_estim = None
            if self.opt.estimate_temp:
                # let's estimate the temperature for the gumbel noise
                h = hidden[0].view(self.opt.layers * self.opt.batch_size * self.opt.rnn_size)
                if self.opt.brnn:
                    h1 = hidden[1].view(self.opt.layers * self.opt.batch_size * self.opt.rnn_size)
                    h = torch.cat([h, h1], 0)
                temp_estim = self.temp_estimator(h.unsqueeze(0)) + 0.5
                self.learned_temp = temp_estim.data[0][0]

            # sample gumbel noise; temp_estim=None in case we don't estimate
            out = self.sampler(out,temp_estim)

        return out


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.generate = False
        self.log = {}
        self.log['distances'] = []

    def set_generate(self, enabled):
        self.generate = enabled

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:, :-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)
        if self.generate:
            out = out.contiguous()
            out = out.view(-1, out.size(2))
            # if estimate temp, then we need to pass the hidden states of the decoder too
            if self.generator.opt.estimate_temp:
                out = self.generator(out, dec_hidden)
            else:
                out = self.generator(out)
        return out, dec_hidden

class D(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.vocab_size = dicts.size()
        self.rnn_size = opt.D_rnn_size
        super(D, self).__init__()
        self.onehot_embedding = nn.Linear(self.vocab_size, self.rnn_size)
        self.rnn1 = nn.LSTM(self.rnn_size, self.rnn_size, 1, bidirectional=True, dropout=opt.D_dropout)
        self.attn = onmt.modules.GlobalAttention(self.rnn_size*2)
        self.l_out = nn.Linear(self.rnn_size * 2, 1)
        if not self.opt.wasser:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        onehot_embeds = self.onehot_embedding(x.contiguous().view(x.size()[0]*x.size()[1], x.size()[2]))
        onehot_embeds = onehot_embeds.view(x.size()[0], x.size()[1], onehot_embeds.size()[1])

        _batch_size = onehot_embeds.size(1)
        h = Variable(torch.zeros(1 * 2, _batch_size, self.rnn_size))
        if self.opt.cuda:
            h = h.cuda()
        c = Variable(torch.zeros(1 * 2, _batch_size, self.rnn_size))
        if self.opt.cuda:
            c = c.cuda()
        outputs, (hn,_) = self.rnn1(onehot_embeds, (h, c))

        hn1 = hn.transpose(0, 1).contiguous().view(_batch_size, -1)
        hn2 = torch.cat([hn[0], hn[1]], 1)
        diff = (hn1 - hn2).norm().data[0]
        assert diff == 0
        out, attn = self.attn(hn2,torch.transpose(outputs,1,0))
        out = self.l_out(out)
        if not self.opt.wasser:
            out = self.sigmoid(out)

        return out