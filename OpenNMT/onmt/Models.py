import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
import torch.nn.functional as F
import numpy as np

_INF = float('inf')
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

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

    def forward(self, input, hidden=None):
        emb = self.word_lut(input)

        if hidden is None:
            batch_size = emb.size(1)
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

        self.layers = []
        for i in range(num_layers):
            layer = nn.LSTMCell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i != len(self.layers):
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts, generator=None):
        self.opt = opt
        self.dicts = dicts
        self.layers = opt.layers
        self.input_feed = opt.input_feed

        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.generator = generator
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        if not self.opt.st_conditioning:
            self.word_lut_unsup = nn.Linear(dicts.size(), opt.word_vec_size)

        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

        self.mask = None

        def applyMask(self, mask):
            self.mask = mask

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)


    def forward(self, input, hidden, context, init_output, H=None, H_crit=None,optimizerH=None, eval=False):

        if self.opt.supervision:
            emb = self.word_lut(input)
            batch_size = input.size(1)

            h_size = (batch_size, self.hidden_size)
            output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

            # n.b. you can increase performance if you compute W_ih * x for all
            # iterations in parallel, but that's only possible if
            # self.input_feed=False
            outputs = []
            output = init_output

            for emb_t in emb.chunk(emb.size(0)):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)
                output, hidden = self.rnn(emb_t, hidden)
                output, attn = self.attn(output, context.t())
                output = self.dropout(output)
                outputs += [output]
            outputs = torch.stack(outputs)
        else:
            if self.opt.st_conditioning:
                emb_t = Variable(torch.LongTensor(1, self.opt.batch_size).zero_().fill_(onmt.Constants.BOS))
            else:
                emb_t = torch.FloatTensor(self.opt.batch_size, self.dicts.size()).zero_()
                emb_t[:, onmt.Constants.BOS] = 1
                emb_t = Variable(emb_t)

            if self.opt.cuda:
                emb_t = emb_t.cuda()

            if self.opt.st_conditioning:
                emb_t = self.word_lut(emb_t)
            else:
                emb_t = self.word_lut_unsup(emb_t)

            outputs = []
            output = init_output

            for i in range(self.opt.max_sent_length):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)

                output, hidden = self.rnn(emb_t, hidden)

                output, attn = self.attn(output, context.t())

                output = self.dropout(output)

                out_t, _ = self.generator(output, hidden)

                # Masking PAD
                out_t.data[:,onmt.Constants.PAD] = 0

                out_t_sofmtmaxed = F.softmax(out_t)
                if self.opt.st_conditioning:
                    pred_t_data = out_t_sofmtmaxed.data.cpu().numpy()
                    argmaxed_preds = np.argmax(pred_t_data, axis=1)
                    argmaxed_preds = torch.from_numpy(argmaxed_preds)
                    argmaxed_preds = Variable(argmaxed_preds)
                    if self.opt.cuda:
                        argmaxed_preds = argmaxed_preds.cuda()
                    emb_t = self.word_lut(argmaxed_preds)
                    if self.opt.batch_size == 1:
                        emb_t = emb_t.unsqueeze(0)
                else:
                    emb_t = self.word_lut_unsup(out_t_sofmtmaxed)

                outputs += [out_t]

            outputs = torch.stack(outputs)
            outputs = outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2))

        return outputs, hidden, attn

class Generator(nn.Module):
    def __init__(self, opt, dicts, temp_estimator=None):
        super(Generator, self).__init__()
        self.opt = opt
        self.dicts = dicts
        self.linear = nn.Linear(opt.rnn_size, dicts.size())
        self.temp_estimator = temp_estimator
        self.tau0 = 1  # initial temperature
        self.eps = 1e-20
        self.temperature = self.tau0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP = 0.5
        self.iter_cnt = 0

    def set_generate(self, enabled):
        self.generate = enabled

    def set_gumbel(self, enabled):
        self.opt.use_gumbel = enabled

    def set_tau(self, val):
        self.tau = val
    def anneal_tau_temp(self):
        # Anneal temperature tau
        self.temperature = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter_cnt * self.opt.batch_size), self.MIN_TEMP)

    def get_noise(self, input):
        noise = torch.rand(input.size())
        if self.opt.cuda:
            noise = noise.cuda()
        noise.add_(self.eps).log_().neg_()
        noise.add_(self.eps).log_().neg_()
        noise = Variable(noise)
        return noise

    def estim_sampler(self, input, temp_estim=None):
        noise = self.get_noise(input)
        x = (input + noise)

        if temp_estim:
            x = x / temp_estim.repeat(x.size())
        else:
            x = x / self.temperature
        return x.view_as(input)

    def sampler(self, input):
        noise = self.get_noise(input)
        x = (input + noise) / self.temperature
        if self.opt.ST:
            # Use ST gumbel-softmax
            y_onehot = torch.FloatTensor(x.size())
            if self.opt.cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.zero_()
            max, idx = torch.max(x, 1)
            y_onehot.scatter_(1, idx.data, 1)
            return Variable(y_onehot).detach()
        else:
            return x.view_as(input)

    def forward(self, input, dec_hidden):
        out = self.linear(input)

        y_onehot = None

        if self.opt.use_gumbel:
            if self.opt.estimate_temp:
                h = dec_hidden[0].view(self.opt.layers * self.opt.batch_size * self.opt.rnn_size)
                if self.opt.brnn:
                    h1 = dec_hidden[1].view(self.opt.layers * self.opt.batch_size * self.opt.rnn_size)
                    h = torch.cat([h, h1], 0)
                temp_estim = self.temp_estimator(h.unsqueeze(0))
                temp_estim = temp_estim + 0.5
                self.temperature = temp_estim.data[0][0]
                out = self.estim_sampler(out, temp_estim)
            else:
                out = self.estim_sampler(out)

        return out, y_onehot

class TempEstimator(nn.Module):
    def __init__(self, opt):
        super(TempEstimator, self).__init__()
        if opt.brnn:
            self.linear1 = nn.Linear(opt.rnn_size*opt.layers*opt.batch_size*2, 1)
        else:
            self.linear1 = nn.Linear(opt.rnn_size * opt.layers * opt.batch_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, input):
        out = self.linear1(input)
        temp = self.softplus(out)

        return temp

class G(nn.Module):

    def __init__(self, opt, encoder, decoder, generator=None, temp_estimator=None):
        super(G, self).__init__()
        self.log = {}
        self.log['distances'] = []
        self.opt = opt
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.generate = False
        self.tau0 = 1 # initial temperature
        self.eps = 1e-20
        self.temperature = self.tau0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP = 0.5
        self.iter_cnt = 0

        # Optionally tie weights as in:
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.opt.tied:
            self.generator.weight = self.encoder.word_lut.weight

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        # h_size = (self.opt.max_sent_length, batch_size, self.decoder.hidden_size)
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


    def forward(self, input, H=None, H_Crit=None, optimizerH=None, eval=False):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output, H, H_Crit,optimizerH, eval)

        if self.opt.supervision:
            out = out.view(-1, out.size(2))
            out = self.generator(out)
            if self.opt.use_gumbel:
                if self.opt.estimate_temp:
                    h = dec_hidden[0].view(self.opt.layers * self.opt.batch_size * self.opt.rnn_size)
                    if self.opt.brnn:
                        h1 = dec_hidden[1].view(self.opt.layers * self.opt.batch_size * self.opt.rnn_size)
                        h = torch.cat([h, h1], 0)
                    temp_estim = self.temp_estimator(h.unsqueeze(0))
                    temp_estim = temp_estim + 0.5
                    self.temperature = temp_estim.data[0][0]
                    out = self.estim_sampler(out, temp_estim)
                else:
                    out = self.estim_sampler(out)
        return out

class D(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.vocab_size = dicts.size()
        self.rnn_size = opt.D_rnn_size
        super(D, self).__init__()
        self.onehot_embedding = nn.Linear(self.vocab_size, self.rnn_size)
        self.rnn1 = nn.LSTM(self.rnn_size, self.rnn_size, 1, bidirectional=True)
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