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
        if not self.opt.st_conditioning:
            self.word_lut_unsup = nn.Linear(dicts.size(), opt.word_vec_size)
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                         opt.word_vec_size,
                                         padding_idx=onmt.Constants.PAD)

        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

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
                out_t = self.generator(output, hidden)

                # Masking PAD
                out_t.data[:,onmt.Constants.PAD] = 0
                out_t_sofmtmaxed = F.softmax(out_t)

                if self.opt.st_conditioning:
                    argmaxed_preds  = torch.max(out_t_sofmtmaxed.data, 1)[1].squeeze()
                    emb_t = self.word_lut(Variable(argmaxed_preds))
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
        self.tau0 = 0.5  # initial temperature
        self.eps = 1e-20
        self.temperature = self.tau0
        self.real_temperature = self.tau0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP = 1
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
        # x = (input + noise) * self.real_temperature
        x = (input + noise) * self.temperature
        return x.view_as(input)

    def forward(self, input, dec_hidden):
        out = self.linear(input)

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

class TempEstimator(nn.Module):
    def __init__(self, opt):
        super(TempEstimator, self).__init__()
        if opt.brnn:
            input_dim = opt.rnn_size*opt.layers*opt.batch_size*2
        else:
            input_dim = opt.rnn_size * opt.layers * opt.batch_size
        linear1_size = round(input_dim * 0.5)
        linear2_size = round(input_dim * 0.5)
        linear3_size = round(input_dim * 0.5)
        linear4_size = round(input_dim * 0.5)
        self.linear1 = nn.Linear(opt.rnn_size*opt.layers*opt.batch_size*2, linear1_size)
        self.linear2 = nn.Linear(linear1_size, linear2_size)
        self.linear3 = nn.Linear(linear2_size, linear3_size)
        self.linear4 = nn.Linear(linear3_size, linear4_size)
        self.linear5 = nn.Linear(linear4_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
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
        return out

class D0(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.vocab_size = dicts.size()
        self.rnn_size = opt.d_rnn_size
        super(D0, self).__init__()
        self.onehot_embedding = nn.Linear(self.vocab_size, self.rnn_size)
        self.rnn1 = nn.LSTM(self.rnn_size, self.rnn_size, 1, bidirectional=True,dropout=opt.d_dropout)
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


class D1(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.vocab_size = dicts.size()
        self.rnn_size = opt.d_rnn_size
        self.num_directions = 2 if opt.brnn else 1
        super(D1, self).__init__()
        self.onehot_embedding = nn.Linear(self.vocab_size, self.rnn_size)
        self.rnn0 = nn.LSTM(self.rnn_size, self.rnn_size,
                        num_layers=opt.d_layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)
        self.l_out = nn.Linear(self.rnn_size * 2, 1)
        if not self.opt.wasser:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        onehot_embeds = self.onehot_embedding(x.contiguous().view(x.size()[0]*x.size()[1], x.size()[2]))
        onehot_embeds = onehot_embeds.view(x.size()[0], x.size()[1], onehot_embeds.size()[1])

        _batch_size = onehot_embeds.size(1)
        h_size = (self.opt.d_layers * self.num_directions, _batch_size, self.rnn_size)
        h_0 = Variable(onehot_embeds.data.new(*h_size).zero_(), requires_grad=False)
        c_0 = Variable(onehot_embeds.data.new(*h_size).zero_(), requires_grad=False)
        hidden = (h_0, c_0)
        onehot_embeds, hidden = self.rnn0(onehot_embeds, hidden)

        out = self.l_out(onehot_embeds.view(x.size()[0]* x.size()[1], onehot_embeds.size()[2]))
        if not self.opt.wasser:
            out = self.sigmoid(out)
        return out


class D3(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.vocab_size = dicts.size()
        self.rnn_size = opt.d_rnn_size
        self.word_vec_size = opt.d_word_vec_size
        self.num_layers = opt.d_layers
        self.num_directions = 2 if opt.brnn else 1
        super(D3, self).__init__()
        self.onehot_embedding = nn.Linear(self.vocab_size, self.word_vec_size)
        self.rnn1 = nn.LSTM(self.word_vec_size, self.rnn_size, num_layers=opt.d_layers, bidirectional=True,dropout=opt.d_dropout)
        self.attn = onmt.modules.GlobalAttention(self.rnn_size*2)
        self.l_out = nn.Linear(self.rnn_size * 2, 1)
        if not self.opt.wasser:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        onehot_embeds = self.onehot_embedding(x.contiguous().view(x.size()[0]*x.size()[1], x.size()[2]))
        embeds = onehot_embeds.view(x.size()[0], x.size()[1], onehot_embeds.size()[1])
        _batch_size = embeds.size(1)

        h = Variable(torch.zeros(self.opt.d_layers * self.num_directions, _batch_size, self.rnn_size))
        if self.opt.cuda:
            h = h.cuda()
        c = Variable(torch.zeros(self.opt.d_layers * self.num_directions, _batch_size, self.rnn_size))
        if self.opt.cuda:
            c = c.cuda()
        context, (hn,_) = self.rnn1(embeds, (h, c))
        hn_tot = torch.cat([hn[0], hn[1]], 1)
        for i in range(2,self.num_layers,2):
            tmp_hn = torch.cat([hn[i], hn[i+1]], 1)
            hn_tot = hn_tot + tmp_hn

        out, attn = self.attn(hn_tot,torch.transpose(context,1,0))
        out = self.l_out(out)
        if not self.opt.wasser:
            out = self.sigmoid(out)
        return out, attn.t()


class CNN(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        super(CNN, self).__init__()
        V = dicts.size()
        D = opt.cnn_embed_dim
        C = 1
        Ci = 1
        Co = opt.cnn_kernel_num
        Ks = opt.cnn_kernel_sizes

        self.embed = nn.Linear(V, D)

        if self.opt.cuda:
            self.convs1 = [nn.Conv2d(Ci, Co, (K, D)).cuda() for K in Ks]
        else:
            self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(opt.cnn_dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        if not self.opt.wasser:
            self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        onehot_embeds = self.embed(x.contiguous().view(x.size()[0] * x.size()[1], x.size()[2]))  # (N,W,D)
        x = onehot_embeds.view(x.size()[0], x.size()[1], onehot_embeds.size()[1])

        x = x.t()

        if self.opt.cnn_static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N,Ci,W,D)

        if self.opt.cuda:
            x = x.cuda()

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        if not self.opt.wasser:
            logit = self.sigmoid(logit)
        return logit, None