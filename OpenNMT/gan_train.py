import onmt
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import time
import numpy as np
import torch.nn.functional as F
# import numba
import logging
from torch import optim
from numpy import random

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
parser.add_argument('-max_sent_length', default=25,
                    help='Maximum sentence length.')

# GAN options
parser.add_argument('-supervision', type=bool, default=False,
                    help='Whether to use supervision')
parser.add_argument('-wasser', type=bool, default=False,
                    help='Use wasserstein optimization')
parser.add_argument('-bgan', type=bool, default=False,
                    help='Use boundary seeking gan')
parser.add_argument('-conditioning_on_gold', type=bool, default=False,
                    help='Use gold for conditioning generation')
parser.add_argument('-st_conditioning', type=bool, default=False,
                    help='Use st for conditioning generation')
parser.add_argument('-hallucinate', type=bool, default=False,
                    help='Whether to use supervision')
parser.add_argument('-perturbe_real', type=bool, default=True,
                    help='Whether to use use gumbel for real data')
parser.add_argument('-h_overfeat', type=int, default=10,
                    help='Overfeat the hallucinator on each batch')
parser.add_argument('-crazy', type=bool, default=False,
                    help='Whether to use supervision')


## G options
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=100,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=100,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=0,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-tied', action='store_true',
                    help='tie the word embedding and softmax weights')
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

parser.add_argument('-estimate_temp', type=bool, default=False,
                    help='Use automatic estimation of temperature annealing for gumbel')

## D options
parser.add_argument('-d_rnn_size', type=int, default=2,
                    help='D: Size fo LSTM hidden states')
parser.add_argument('-d_dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-d_layers', type=int, default=3,
                    help='Number of layers in the LSTM encoder/decoder')

# ## Hallucinator options
# parser.add_argument('-H_rnn_size', type=int, default=20,
#                     help='D: Size fo LSTM hidden states')
# parser.add_argument('-H_dropout', type=float, default=0.3,
#                     help='Dropout probability; applied between LSTM stacks.')


## G Optimization options
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=10000000,
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


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit


def H_memoryEfficientLoss(H , dataset, outputs, sources, targets, generator, crit, log_pred=False, eval=False):
    # compute generations one piece at a time
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for out_t, targ_t in zip(outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        out_t = generator(out_t)
        pred_t = F.log_softmax(out_t)
        if log_pred:
            log_predictions(pred_t, sources, targets, H.log['distances'], dataset['dicts']['tgt'])
        loss_t = crit(pred_t, targ_t.view(-1))
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output

def log(G, outputs, sources, targets, dataset):

        pred_t = F.log_softmax(outputs)
        log_predictions(pred_t, sources, targets, G.log['distances'], dataset['dicts']['tgt'])


def lev_dist(source, target):

    # @numba.jit("f4(i8[:], i8[:])", nopython=True, cache=True, target="cpu")
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


def log_predictions(pred_t, src_t, targ_t, distances, tgt_dict):
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

    argmax_inputs = src_t.data.cpu().numpy()
    argmax_sources = np.ones((argmax_inputs[0].size, len(argmax_inputs)))

    for i in range(argmax_inputs[0].size):
        for j in range(len(argmax_inputs)):
            argmax_sources[i][j] = argmax_inputs[j][i].astype(int)

    argmax_preds_sorted = argmax_preds_sorted.astype(int)
    rand_idx = np.random.randint(len(argmax_preds_sorted))
    logger.debug('SAMPLE:')
    logger.debug('source: ' + str(" ".join(tgt_dict.convertToLabels(argmax_sources[rand_idx], onmt.Constants.EOS))))
    logger.debug('preds: ' + str(" ".join(tgt_dict.convertToLabels(argmax_preds_sorted[rand_idx], onmt.Constants.EOS))))
    logger.debug('trgts: ' + str(" ".join(tgt_dict.convertToLabels(argmax_targets[rand_idx].astype(int), onmt.Constants.EOS))))
    distances.append(lev_dist(argmax_targets[rand_idx].astype(int), argmax_preds_sorted[rand_idx]))
    if len(distances) <= 10:
        avg_dist = np.mean(distances)
        avg_dist_10 = avg_dist
    else:
        avg_dist = np.mean(distances[:-10])
        avg_dist_10 = np.mean(distances[-10:])
    logger.debug('past avg lev distance: %f, last 10 avg lev distance %f\n' % (avg_dist, avg_dist_10))



def trainModel(G, trainData, validData, dataset, optimizerG, H1=None, H2=None,
               H_crit=None, optimizerH1=None, optimizerH2=None,
               CRAZY=None, optimizerCRAZY=None):
    logger.info(G)
    G.train()

    if opt.hallucinate:
        logger.info(H1)
        H1.train()

        logger.info(H2)
        H2.train()

    if opt.crazy:
        logger.info(CRAZY)
        CRAZY.train()


    cxt_criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    def trainEpoch(epoch):

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        real_report_loss = real_report_words = 0
        fake_report_loss = fake_report_words = 0

        start = time.time()

        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i
            batch = trainData[batchIdx]
            outputs = G(batch, H1, H_crit, optimizerH1, eval=False)
            sources = batch[0]
            targets = batch[1][1:]  # exclude <s> from targets


            log_pred = i % (opt.log_interval) == 0 and i > 0

            if opt.hallucinate:
                total_loss, report_loss = 0, 0
                total_words, report_words = 0, 0
                for j in range(opt.h_overfeat):
                    H1.zero_grad()
                    h_outputs = H1(batch)
                    sources = batch[0]
                    targets = batch[1][1:]  # exclude <s> from targets

                    if log_pred and j == 4:
                        logger.debug("[HALLUCINATOR 1]:")
                    loss, gradOutput = H_memoryEfficientLoss(
                        H1, dataset,h_outputs, sources, targets, H1.generator, cxt_criterion, log_pred and j == opt.h_overfeat-1)
                    h_outputs.backward(gradOutput)


                    # update the parameters
                    grad_norm = optimizerH1.step()

                    report_loss += loss
                    total_loss += loss
                    num_words = targets.data.ne(onmt.Constants.PAD).sum()
                    total_words += num_words
                    report_words += num_words

                    if i % opt.log_interval == 0 and i > 0 and j == opt.h_overfeat-1:
                        logger.debug("Epoch %2d, %5d/%5d batches; perplexity: %6.2f; %3.0f tokens/s\n'" %
                              (epoch, i, len(trainData),
                               math.exp(report_loss / report_words),
                               report_words / (time.time() - start)))
                    report_loss = report_words = 0

                h_outputs = H1(batch)
                h_outputs = h_outputs.view(-1, h_outputs.size(2))
                hallucination = H1.generator(h_outputs)

                total_loss, report_loss = 0, 0
                total_words, report_words = 0, 0
                for j in range(opt.h_overfeat):
                    H2.zero_grad()
                    inverse_sources = batch[1][1:-1]
                    inverse_targets = batch[0]
                    bos = torch.LongTensor(1, inverse_targets.size(1)).fill_(onmt.Constants.BOS)
                    eos = torch.LongTensor(1, inverse_targets.size(1)).fill_(onmt.Constants.EOS)
                    if opt.cuda:
                        eos = eos.cuda()
                        bos = bos.cuda()
                    inverse_targets = Variable(torch.cat([bos,inverse_targets.data],0))
                    inverse_targets = Variable(torch.cat([inverse_targets.data,eos],0))
                    h_outputs = H2((inverse_sources, inverse_targets))
                    inverse_targets = inverse_targets[1:]
                    if log_pred and j == opt.h_overfeat-1:
                        logger.debug("[HALLUCINATOR 2]:")
                    loss, gradOutput = H_memoryEfficientLoss(
                        H2, dataset,h_outputs, inverse_sources, inverse_targets, H2.generator, cxt_criterion, log_pred and j == opt.h_overfeat-1)
                    h_outputs.backward(gradOutput)


                    # update the parameters
                    grad_norm = optimizerH2.step()

                    report_loss += loss
                    total_loss += loss
                    num_words = inverse_targets.data.ne(onmt.Constants.PAD).sum()
                    total_words += num_words
                    report_words += num_words
                    if i % opt.log_interval == 0 and i > 0 and j == opt.h_overfeat-1:
                        logger.debug("Epoch %2d, %5d/%5d batches; perplexity: %6.2f; %3.0f tokens/s\n'" %
                              (epoch, i, len(trainData),
                               math.exp(report_loss / report_words),
                               report_words / (time.time() - start)))
                    report_loss = report_words = 0

                h_outputs = H2((inverse_sources, inverse_targets))
                h_outputs = h_outputs.view(-1, h_outputs.size(2))
                inverse_hallucination = H2.generator(h_outputs)

            if log_pred:
                logger.debug("[GENERATOR]:")
                log(G, outputs, sources, targets, dataset)



            pert1 = hallucination
            if opt.perturbe_real:
                pert1 = G.generator.sampler(hallucination)
            # Masking PAD: we do it before softmax, as in generation
            pert1.data[:, onmt.Constants.PAD] = 0

            hallucination = F.log_softmax(pert1)

            pert2 = inverse_hallucination

            if opt.perturbe_real:
                pert2 = G.generator.sampler(inverse_hallucination)
            # Masking PAD: we do it before softmax, as in generation
            pert2.data[:, onmt.Constants.PAD] = 0
            inverse_hallucination = F.log_softmax(pert2)

            # LEARNING FAKE

            G.zero_grad()
            pred_t = F.log_softmax(outputs)
            fake_batch = (pred_t, inverse_hallucination)
            D_fake = CRAZY(fake_batch)

            D_G_z1 = D_fake.data.mean()
            D_G_z2 = D_G_z1

            inverse_hallucination = Variable(inverse_hallucination.data, requires_grad=False)

            # hallucination_view = hallucination.view(hallucination.size(0)/opt.batch_size,opt.batch_size,hallucination.size(1))
            # inverse_hallucination_view = inverse_hallucination.view(inverse_hallucination.size(0)/opt.batch_size,opt.batch_size,inverse_hallucination.size(1))
            # print('inverse_hallucination: ' + str(inverse_hallucination))
            sources = Variable(torch.max(hallucination.data, 1)[1].squeeze(), requires_grad=False)
            targets = Variable(torch.max(inverse_hallucination.data, 1)[1].squeeze(), requires_grad=False)
            # sources = sources.view(sources.size(0)/opt.batch_size,opt.batch_size)
            sources = sources.view(sources.size(0)/opt.batch_size, opt.batch_size)
            targets = targets.view(targets.size(0)/opt.batch_size, opt.batch_size)
            num_words = targets.data.ne(onmt.Constants.PAD).sum()

            real_report_words += num_words
            fake_report_words += num_words

            # D_fake = D_fake.view(D_fake.size(0)/opt.batch_size,opt.batch_size,D_fake.size(1))
            if log_pred:
                logger.debug("[FAKE]:")
            loss, gradOutput = H_memoryEfficientLoss(
                CRAZY, dataset, D_fake, sources, targets, CRAZY.generator, cxt_criterion,
                log_pred)
            D_fake.backward(gradOutput)
            fake_report_loss += loss
            optimizerG.step()

            # errG = cxt_criterion(D_fake, labels)
            # errG.backward()

            # print('ITERATION: ')
            # for p in G.parameters():
            #     print('p.grad.data: ' + str(p.grad.data))


            # LEARNING REAL
            CRAZY.zero_grad()
            real_batch = (hallucination.detach(), inverse_hallucination)

            D_real = CRAZY(real_batch)
            D_x = D_real.data.mean()
            if log_pred:
                logger.debug("[REAL]:")
            loss, gradOutput = H_memoryEfficientLoss(
                CRAZY, dataset, D_real, sources, targets, CRAZY.generator, cxt_criterion,
                log_pred)

            D_real.backward(gradOutput)
            real_report_loss += loss
            optimizerCRAZY.step()
            # errD = cxt_criterion(D_real, labels)
            # real_report_loss += errD.data[0]
            # errD.backward()

            # print('ITERATION: ')
            # for p in CRAZY.parameters():
            #     print('p.grad.data: ' + str(p.grad.data))

            # anneal tau for gumbel
            if opt.use_gumbel and opt.gumbel_anneal_interval > 0 and not opt.estimate_temp and i % opt.gumbel_anneal_interval == 0 and i > 0:
                G.anneal_tau_temp()

            if i % opt.log_interval == 0 and i > 0:
                # log_crazy(D_fake, D_real, hallucination, inverse_hallucination, pred_t)
                logger.info('[%d/%d][%d/%d] Temp: %.4f; Real Ppl: %6.2f, Fake Ppl: %6.2f;\n\n'
                      % (epoch, opt.epochs, i, len(trainData),
                         G.generator.temperature,
                         math.exp(real_report_loss / real_report_words), math.exp(fake_report_loss / fake_report_words)))

                real_report_loss = real_report_words = 0
                fake_report_loss = fake_report_words = 0
            start = time.time()
            G.iter_cnt+=1

        return 0

    def log_crazy(D_fake, D_real, hallucination, inverse_hallucination, pred_t):

        # argmax conversion

        # REAL
        # argmax_hallucination_sorted = get_crazy_argmax(hallucination)
        # argmax_inverse_hallucination_sorted = get_crazy_argmax(inverse_hallucination)
        # argmax_dreal_sorted = get_crazy_argmax(D_real)
        #
        # FAKE
        # argmax_preds_sorted = get_crazy_argmax(pred_t)
        # argmax_dfake_sorted = get_crazy_argmax(D_fake)


        # radnomly sample one sentence
        rand_idx = np.random.randint(len(argmax_preds_sorted))
        logger.debug("[CRAZY]:")
        logger.debug('SAMPLE:')
        logger.debug(
            'hallucinated targets: ' + str(" ".join(
                dataset['dicts']['tgt'].convertToLabels(argmax_hallucination_sorted[rand_idx], onmt.Constants.EOS))))
        logger.debug(
            'hallucinated sources: ' + str(" ".join(
                dataset['dicts']['tgt'].convertToLabels(argmax_inverse_hallucination_sorted[rand_idx],
                                                        onmt.Constants.EOS))))
        logger.debug(
            'crazy real sources:   ' + str(" ".join(
                dataset['dicts']['tgt'].convertToLabels(argmax_dreal_sorted[rand_idx],
                                                        onmt.Constants.EOS))))
        logger.debug(
            'generated targets:    ' + str(
                " ".join(dataset['dicts']['tgt'].convertToLabels(argmax_preds_sorted[rand_idx], onmt.Constants.EOS))))
        logger.debug(
            'generated sources:    ' + str(
                " ".join(dataset['dicts']['tgt'].convertToLabels(argmax_dfake_sorted[rand_idx], onmt.Constants.EOS))))



    for epoch in range(opt.start_epoch, opt.epochs + 1):

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        logger.info('Semi-supervision train loss: %g' % train_loss)

        # valid_loss = eval(G, cxt_criterion, validData, dataset)
        # valid_ppl = math.exp(min(valid_loss, 100))
        # logger.info('Validation perplexity: %g' % valid_ppl)

        #  (4) drop a checkpoint
        checkpoint = {
            'model': G,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optimizerG,
        }
        torch.save(checkpoint,
                   '%s_e%d.pt' % (opt.save_model, epoch))


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
        temp_estimator = None
        if opt.supervision:
            decoder = onmt.Models.Decoder(opt, dicts['tgt'])
            generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()))
            if opt.estimate_temp:
                temp_estimator = onmt.Models.TempEstimator(opt)
        else:
            temp_estimator = None
            if opt.estimate_temp:
                temp_estimator = onmt.Models.TempEstimator(opt)
            H1 = None
            H_crit= None
            if opt.hallucinate:
                h_encoder = onmt.Hallucinator.H_Encoder(opt, dicts['src'])
                h_decoder = onmt.Hallucinator.H_Decoder(opt, dicts['tgt'])
                h_generator = nn.Sequential(
                    nn.Linear(opt.rnn_size, dicts['tgt'].size()))
                H1 = onmt.Hallucinator.Hallucinator(opt, h_encoder, h_decoder, h_generator)


                h_encoder = onmt.Hallucinator.H_Encoder(opt, dicts['src'])
                h_decoder = onmt.Hallucinator.H_Decoder(opt, dicts['tgt'])
                h_generator = nn.Sequential(
                    nn.Linear(opt.rnn_size, dicts['tgt'].size()))
                H2 = onmt.Hallucinator.Hallucinator(opt, h_encoder, h_decoder, h_generator)

                for p in H1.parameters():
                    p.data.uniform_(-opt.param_init, opt.param_init)
                optimizerH1 = optim.Adam(H1.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
                # optimizerH1 = optim.Adam(H1.parameters(), lr=2e-2, betas=(opt.beta1, 0.999))

                for p in H2.parameters():
                    p.data.uniform_(-opt.param_init, opt.param_init)
                optimizerH2 = optim.Adam(H2.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
                # optimizerH2 = optim.Adam(H2.parameters(), lr=2e-2, betas=(opt.beta1, 0.999))

            generator = onmt.Models.Generator(opt, dicts['tgt'], temp_estimator)
            decoder = onmt.Models.Decoder(opt, dicts['tgt'], generator)

            if opt.crazy:
                h_encoder = onmt.S2SDisc.H_Encoder(opt, dicts['src'])
                h_decoder = onmt.S2SDisc.H_Decoder(opt, dicts['tgt'])
                h_generator = nn.Sequential(
                    nn.Linear(opt.rnn_size, dicts['tgt'].size()))
                CRAZY = onmt.S2SDisc.CRAZY(opt, h_encoder, h_decoder, h_generator)
                for p in CRAZY.parameters():
                    p.data.uniform_(-opt.param_init, opt.param_init)
                optimizerCRAZY = optim.Adam(CRAZY.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))


        G = onmt.Models.G(opt, encoder, decoder, generator, temp_estimator)
        for p in G.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optimizerG = optim.Adam(G.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))


    else:
        logger.info('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        G = checkpoint['G']
        optimizerG = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch']

    if opt.cuda:
        G.cuda()
        if not opt.supervision:
            if opt.hallucinate:
                H1.cuda()
                H2.cuda()
            if opt.crazy:
                CRAZY.cuda()
    else:
        G.cpu()
        if not opt.supervision:
            if opt.hallucinate:
                H1.cpu()
                H2.cpu()
            if opt.crazy:
                CRAZY.cpu()

    nParams = sum([p.nelement() for p in G.parameters()])
    logger.info('* number of G parameters: %d' % nParams)
    nParams = sum([p.nelement() for p in H1.parameters()])
    logger.info('* number of H parameters: %d' % nParams)
    nParams = sum([p.nelement() for p in CRAZY.parameters()])
    logger.info('* number of CRAZY parameters: %d' % nParams)
    trainModel(G, trainData, validData, dataset, optimizerG, H1, H2, H_crit, optimizerH1, optimizerH2, CRAZY, optimizerCRAZY)


if __name__ == "__main__":
    main()