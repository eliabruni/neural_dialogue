import onmt
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import time
import numpy as np
import torch.nn.functional as F
import numba
import logging
from torch import optim

parser = argparse.ArgumentParser(description='gan_train.py')

## Data options
parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from',
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## G options
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', type=bool, default=False,
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-use_gumbel', type=bool, default=False,
                    help='Use gumbel softmax')

parser.add_argument('-gumbel_anneal_interval', type=int, default=-1,
                    help="""Temperature annealing interval for gumbel. -1 to switch
                         off the annealing""")
parser.add_argument('-ST', type=bool, default=False,
                    help='ST gumbel softmax')

parser.add_argument('-estimate_temp', type=bool, default=False,
                    help='Use automatic estimation of temperature annealing for gumbel')

## D options
parser.add_argument('-D_rnn_size', type=int, default=500,
                    help='D: Size fo LSTM hidden states')

## G Optimization options
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('-batch_size', type=int, default=1,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=1000,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('-learning_rate', type=float, default=2e-4,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=20,
                    help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-cuda', action='store_true',
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
parser.add_argument('-seed', type=int, default=1111,
                    help="Seed for random initialization")

opt = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(opt.seed)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(opt)
if torch.cuda.is_available():
    if not opt.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(opt.seed)


def getBatch(outputs, dec_hidden, sources, targets, G, dataset, log_pred=False, eval=False):
    # compute generations one piece at a time

    if opt.estimate_temp:
        # let's estimate the temperature for the gumbel noise
        h = dec_hidden[0].view(G.opt.layers * G.opt.batch_size * opt.rnn_size)
        if G.opt.brnn:
            h1 = dec_hidden[1].view(G.opt.layers * G.opt.batch_size * opt.rnn_size)
            h = torch.cat([h,h1],0)
        temp_estim = G.temp_estimator(h.unsqueeze(0))
        temp_estim = temp_estim + 0.5
        G.temperature = temp_estim.data[0][0]

        temp_estim_gen = Variable(temp_estim.data, requires_grad=(not eval), volatile=eval)

    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    sources_split = torch.split(sources, opt.max_generator_batches)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for src_t, out_t, targ_t in zip(sources_split, outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        out_t = G.generator(out_t)
        if opt.use_gumbel:
            if opt.estimate_temp:
                pred_t = G.estim_sampler(out_t, temp_estim_gen)
            else:
                pred_t = G.sampler(out_t)
        else:
            pred_t = F.log_softmax(out_t)

        if log_pred:
            log_predictions(pred_t, targ_t, G.log['distances'])

        noise_sources = one_hot(G, src_t.data,
                                     dataset['dicts']['src'].size())
        noise_targets = one_hot(G, targ_t.data,
                                     dataset['dicts']['tgt'].size())

        if opt.cuda:
            noise_sources = noise_sources.cuda()
            noise_targets = noise_targets.cuda()
            pred_t = pred_t.cuda()
        fake = torch.cat([noise_sources, pred_t], 0)
        real = torch.cat([noise_sources,noise_targets],0)

    grad_output = None if outputs.grad is None else outputs.grad.data
    return fake, real, grad_output
    # return pred_t, noise_targets, grad_output


def lev_dist(source, target):

    @numba.jit("f4(i8[:], i8[:])", nopython=True, cache=True, target="cpu")
    def jitted_lev_dist(vec1, vec2):
        # Prepare a matrix
        dist = np.zeros((vec1.size + 1, vec2.size + 1))
        dist[:, 0] = np.arange(vec1.size + 1)
        dist[0, :] = np.arange(vec2.size + 1)

        # Compute distance
        for i in xrange(vec1.size):
            for j in xrange(vec2.size):
                cost = 0 if vec1[i] == vec2[j] else 1
                dist[i + 1, j + 1] = min(
                    dist[i, j + 1] + 1,   # deletion
                    dist[i + 1, j] + 1,   # insertion
                    dist[i, j] + cost   # substitution
                )
        return dist[-1][-1] / max(vec1.size, vec2.size)

    return 0 if np.array_equal(source, target) else jitted_lev_dist(source, target)


def log_predictions(pred_t, targ_t, distances):
    pred_t_data = pred_t.data.cpu().numpy()
    argmaxed_preds = np.argmax(pred_t_data, axis=1)
    argmax_preds_sorted = np.ones((opt.batch_size,argmaxed_preds.size/opt.batch_size ))
    cnt=0
    for i in range(0, argmaxed_preds.size, opt.batch_size):
        for j in range(opt.batch_size):
            argmax_preds_sorted[j][cnt] = argmaxed_preds[i+j]
        cnt+=1
    argmax_inputs = targ_t.data.cpu().numpy()
    argmax_targets = np.ones((argmax_inputs[0].size, len(argmax_inputs)))

    for i in range(argmax_inputs[0].size):
        for j in range(len(argmax_inputs)):
            argmax_targets[i][j] = argmax_inputs[j][i].astype(int)

    argmax_preds_sorted = argmax_preds_sorted.astype(int)
    rand_idx = np.random.randint(len(argmax_preds_sorted))
    logger.debug('SAMPLE:')
    logger.debug('preds: ' + str(argmax_preds_sorted[rand_idx]))
    logger.debug('trgts: ' + str(argmax_targets[rand_idx].astype(int)))
    distances.append(lev_dist(argmax_targets[rand_idx].astype(int), argmax_preds_sorted[rand_idx]))
    if len(distances) <= 10:
        avg_dist = np.mean(distances)
        avg_dist_10 = avg_dist
    else:
        avg_dist = np.mean(distances[:-10])
        avg_dist_10 = np.mean(distances[-10:])
    logger.debug('past avg lev distance: %f, last 10 avg lev distance %f \n' % (avg_dist, avg_dist_10))


def one_hot(G, input, num_input_symbols, temp_estim=None):
    one_hot_tensor = torch.FloatTensor(input.size()[1], input.size()[0], num_input_symbols)
    input = torch.transpose(input, 1, 0)
    for i in range(input.size()[0]):
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(input.size()[1], num_input_symbols)
        if opt.cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, input[i].unsqueeze(1),5)
        if opt.ST:
            # Use ST gumbel-softmax
            one_hot_tensor[i] = y_onehot
        else:
            # Use soft gumbel-softmax
            pert = G.sampler(Variable(y_onehot))
            one_hot_tensor[i] = pert.data

    one_hot_tensor = torch.transpose(one_hot_tensor,1,0)
    return Variable(one_hot_tensor.contiguous().view(one_hot_tensor.size()[0]*one_hot_tensor.size()[1], one_hot_tensor.size()[2]))


def clip_gradient(opt, model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, opt.clip / (totalnorm + 1e-6))


def trainModel(G, D, trainData, validData, dataset, optimizerG, optimizerD):
    logger.info(G)
    logger.info(D)
    G.train()

    # define criterion of each GPU
    criterion = nn.BCELoss()

    # GAN variables
    real_label = 1
    fake_label = 0

    def trainEpoch(epoch):

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i
            batch = trainData[batchIdx]

            outputs, dec_hidden = G(batch)
            sources = batch[0]
            targets = batch[1][1:]  # exclude <s> from targets
            log_pred = i % (opt.log_interval) == 0 and i > 0
            fake, real, gradOutput = getBatch(
                    outputs, dec_hidden, sources, targets, G, dataset, log_pred)

            fake = fake.contiguous().view(fake.size()[0]/opt.batch_size,opt.batch_size,fake.size()[1])
            real = real.contiguous().view(real.size()[0]/opt.batch_size,opt.batch_size,real.size()[1])

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            D.zero_grad()

            # train with real
            output = D(real)
            label = torch.FloatTensor(opt.batch_size)
            if opt.cuda:
                label = label.cuda()
            label = Variable(label)
            label.data.resize_(output.size()[0]).fill_(real_label)
            label = label.unsqueeze(1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            label.data.fill_(fake_label)
            output = D(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake

            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()

            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = D(fake)
            errG = criterion(output, label)

            errG.backward()
            outputs.backward(gradOutput)
            # print('')
            # print('G grads:')
            # print('ITERATION: ')
            # for p in G.parameters():
            #     print('p.grad.data: ' + str(p.grad.data))
            # print('XXXXXXXXX')
            # print('XXXXXXXXX')
            # print('XXXXXXXXX')
            # print('XXXXXXXXX')
            D_G_z2 = output.data.mean()
            optimizerG.step()

            # anneal tau for gumbel
            if opt.use_gumbel and opt.gumbel_anneal_interval > 0 and not opt.estimate_temp and i % opt.gumbel_anneal_interval == 0 and i > 0:
                G.anneal_tau_temp()

            if i % opt.log_interval == 0 and i > 0:
                logger.info('[%d/%d][%d/%d] Temp: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.epochs, i, len(trainData),
                         G.temperature, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

                report_loss = report_words = 0
                start = time.time()
            G.iter_cnt+=1
        return total_loss / i

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        logger.info('Semi-supervision train loss: %g' % train_loss)


def main():

    logger.info("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda)

    dicts = dataset['dicts']
    logger.info(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building G...')

    if opt.train_from is None:
        encoder = onmt.Models.Encoder(opt, dicts['src'])
        decoder = onmt.Models.Decoder(opt, dicts['tgt'])
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()))
        if opt.estimate_temp:
            temp_estimator = onmt.Models.TempEstimator(opt)
            G = onmt.Models.G(opt, encoder, decoder, generator, temp_estimator)
        else:
            G = onmt.Models.G(opt, encoder, decoder, generator)
        D = onmt.Models.D(opt, dicts['tgt'])

        # optimizerG = onmt.Optim(
        #     G.parameters(), opt.optim, 1, opt.max_grad_norm,
        #     lr_decay=0.5,
        #     start_decay_at=opt.start_decay_at
        # )
        #
        # optimizerD = onmt.Optim(
        #     D.parameters(), opt.optim, 1, opt.max_grad_norm,
        #     lr_decay=0.5,
        #     start_decay_at=opt.start_decay_at
        # )

        optimizerG = optim.Adam(G.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        optimizerD = optim.Adam(D.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))

    else:
        logger.info('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        G = checkpoint['G']
        optimizerG = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch']

    if opt.cuda:
        G.cuda()
        D.cuda()
    else:
        G.cpu()
        D.cpu()

    nParams = sum([p.nelement() for p in G.parameters()])
    logger.info('* number of G parameters: %d' % nParams)
    nParams = sum([p.nelement() for p in D.parameters()])
    logger.info('* number of D parameters: %d' % nParams)
    trainModel(G, D, trainData, validData, dataset, optimizerG, optimizerD)


if __name__ == "__main__":
    main()