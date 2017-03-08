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

# GAN options
parser.add_argument('-supervision', type=bool, default=False,
                    help='Whether to use supervision')
parser.add_argument('-wasser', type=bool, default=False,
                    help='Use wasserstein optimization')

## G options
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=10,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=10,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=0,
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
parser.add_argument('-D_rnn_size', type=int, default=10,
                    help='D: Size fo LSTM hidden states')
parser.add_argument('-D_dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')


## G Optimization options
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=64,
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

parser.add_argument('-log_interval', type=int, default=10,
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


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit

def eval(G, criterion, data, dataset):
    total_loss = 0
    total_words = 0

    G.eval()
    if opt.use_gumbel:
        G.set_gumbel(False)
    for i in range(len(data)):
        batch = data[i] # must be batch first for gather/scatter in DataParallel
        outputs = G(batch)
        sources = batch[0]
        targets = batch[1][1:]  # exclude <s> from targets
        log_pred = i % (opt.log_interval) == 0 and i > 0
        _, _, loss, _,  = memoryEfficientLoss(G, outputs, sources, targets, dataset, criterion, log_pred, True)

        total_loss += loss
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    G.train()
    if opt.use_gumbel:
        G.set_gumbel(True)
    return total_loss / total_words

def memoryEfficientLoss(G, outputs, sources, targets, dataset, criterion, log_pred=False, eval=False):

    grad_output = None
    loss = 0
    fake, real = None, None

    if opt.supervision or eval:
        pred_t = F.log_softmax(outputs)

        if log_pred:
            log_predictions(pred_t, targets, G.log['distances'], dataset['dicts']['tgt'])
        targ_t = targets.contiguous()
        loss_t = criterion(pred_t, targ_t.view(-1))
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(opt.batch_size).backward()

        grad_output = None if outputs.grad is None else outputs.grad.data

    else:
        pred_t = F.softmax(outputs)

        if log_pred:
            log_predictions(pred_t, targets, G.log['distances'], dataset['dicts']['tgt'])

        noise_sources = one_hot(G, sources.data,
                                     dataset['dicts']['src'].size())
        noise_targets = one_hot(G, targets.data,
                                     dataset['dicts']['tgt'].size())

        if opt.cuda:
            noise_sources = noise_sources.cuda()
            noise_targets = noise_targets.cuda()
            pred_t = pred_t.cuda()
        fake = torch.cat([noise_sources, pred_t], 0)
        real = torch.cat([noise_sources,noise_targets],0)

    return fake, real, loss, grad_output


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


def log_predictions(pred_t, targ_t, distances, tgt_dict):
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
    logger.debug('preds: ' + str(" ".join(tgt_dict.convertToLabels(argmax_preds_sorted[rand_idx], onmt.Constants.EOS))))
    logger.debug('trgts: ' + str(" ".join(tgt_dict.convertToLabels(argmax_targets[rand_idx].astype(int), onmt.Constants.EOS))))
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

        if opt.ST:
            # Use ST gumbel-softmax
            y_onehot.scatter_(1, input[i].unsqueeze(1), 1)
            one_hot_tensor[i] = y_onehot
        else:
            # Use soft gumbel-softmax
            y_onehot.scatter_(1, input[i].unsqueeze(1), num_input_symbols)
            pert = G.sampler(Variable(y_onehot))
            pert = F.softmax(pert)
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

def trainModel(G, trainData, validData, dataset, optimizerG, D=None, optimizerD=None):
    logger.info(G)
    G.train()
    if not opt.supervision:
        logger.info(D)
        D.train()

    # define criterion of each GPU
    criterion = nn.BCELoss()

    cxt_criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    # GAN variables
    real_label = 1
    fake_label = 0
    start_time = time.time()
    def trainEpoch(epoch):

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i
            batch = trainData[batchIdx]

            outputs = G(batch)
            sources = batch[0]
            targets = batch[1][1:]  # exclude <s> from targets
            log_pred = i % (opt.log_interval*5) == 0 and i > 0

            if opt.supervision:
                fake, real, loss, gradOutput = memoryEfficientLoss(
                    G, outputs, sources, targets, dataset, cxt_criterion, log_pred)

                # update the parameters
                grad_norm = optimizerG.step()
                report_loss += loss
                total_loss += loss
                num_words = targets.data.ne(onmt.Constants.PAD).sum()
                total_words += num_words
                report_words += num_words
                if i % opt.log_interval == 0 and i > 0:
                    print("Epoch %2d, %5d/%5d batches; perplexity: %6.2f; %3.0f tokens/s; %6.0f s elapsed" %
                          (epoch, i, len(trainData),
                           math.exp(report_loss / report_words),
                           report_words / (time.time() - start),
                           time.time() - start_time))
                    if opt.use_gumbel:
                        learned_temp = G.temperature
                        print("Generated temp: %.4f " % (learned_temp))
                        # print("Real temp: %.4f, Generated temp: %.4f " % (G.generator.real_temp, learned_temp))
            else:

                fake, real, _, _ = memoryEfficientLoss(
                    G, outputs, sources, targets, dataset, None, log_pred)

                fake = fake.contiguous().view(fake.size()[0]/opt.batch_size,opt.batch_size,fake.size()[1])
                real = real.contiguous().view(real.size()[0]/opt.batch_size,opt.batch_size,real.size()[1])

                G_train_interval = 10
                if opt.wasser:
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    D.zero_grad()

                    # train with real
                    D_real = D(real)
                    D_x = D_real.data.mean()

                    # train with fake
                    D_fake = D(fake.detach())

                    D_G_z1 = D_fake.data.mean()

                    errD = -(torch.mean(D_real) - torch.mean(D_fake))
                    errD.backward()

                    optimizerD.step()

                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    if i % G_train_interval == 0:

                        # if i % G_train_interval == 0:
                        ############################
                        # (2) Update G network: maximize log(D(G(z)))
                        ###########################
                        G.zero_grad()

                        D_fake = D(fake)
                        errG = -torch.mean(D_fake)

                        errG.backward()

                        D_G_z2 = D_fake.data.mean()

                        # print('ITERATION: ')
                        # for p in G.parameters():
                        #     print('p.grad.data: ' + str(p.grad.data))

                        optimizerG.step()

                else:
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

                    D_G_z2 = output.data.mean()

                    # print('ITERATION: ')
                    # for p in G.parameters():
                    #     print('p.grad.data: ' + str(p.grad.data))

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

        if epoch == 1:
            valid_loss = eval(G, cxt_criterion, validData, dataset)
            valid_ppl = math.exp(min(valid_loss, 100))
            logger.info('Initial validation perplexity: %g' % valid_ppl)

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        logger.info('Semi-supervision train loss: %g' % train_loss)

        valid_loss = eval(G, cxt_criterion, validData, dataset)
        valid_ppl = math.exp(min(valid_loss, 100))
        logger.info('Validation perplexity: %g' % valid_ppl)


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
        temp_estimator = None
        if opt.estimate_temp:
            temp_estimator = onmt.Models.TempEstimator(opt)

        G = onmt.Models.G(opt, encoder, decoder, generator, temp_estimator)
        G.set_generate(True)
        for p in G.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optimizerG = optim.Adam(G.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))

        D = None
        optimizerD = None
        if not opt.supervision:
            D = onmt.Models.D(opt, dicts['tgt'])
            for p in D.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)
            if opt.wasser:
                optimizerG = optim.RMSprop(G.parameters(), lr=5e-5)
                optimizerD = optim.RMSprop(D.parameters(), lr=5e-5)
            else:
                optimizerD = optim.Adam(D.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))


    else:
        logger.info('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        G = checkpoint['G']
        optimizerG = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch']

    if opt.cuda:
        G.cuda()
        if not opt.supervision:
            D.cuda()
    else:
        G.cpu()
        if not opt.supervision:
            D.cpu()

    nParams = sum([p.nelement() for p in G.parameters()])
    logger.info('* number of G parameters: %d' % nParams)
    if not opt.supervision:
        nParams = sum([p.nelement() for p in D.parameters()])
        logger.info('* number of D parameters: %d' % nParams)
    trainModel(G, trainData, validData, dataset, optimizerG, D, optimizerD)



if __name__ == "__main__":
    main()