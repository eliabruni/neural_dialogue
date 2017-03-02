import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
from torch import optim

parser = argparse.ArgumentParser(description='train.py')

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


## GAN options
parser.add_argument('-generate', type=bool, default=True,
                    help='Whether to generate')
parser.add_argument('-supervision', type=bool, default=False,
                    help='Whether to use supervision')
parser.add_argument('-use_gumbel', type=bool, default=False,
                    help='Use gumbel softmax')
parser.add_argument('-ST', type=bool, default=False,
                    help='ST gumbel softmax')
parser.add_argument('-wasser', type=bool, default=False,
                    help='Use wasserstein optimization')
parser.add_argument('-estimate_temp', type=bool, default=False,
                    help='Use automatic estimation of temperature annealing for gumbel')
parser.add_argument('-gumbel_anneal_interval', type=int, default=1000,
                    help="""Temperature annealing interval for gumbel. -1 to switch
                         off the annealing""")
## D options
parser.add_argument('-D_rnn_size', type=int, default=500,
                    help='D: Size fo LSTM hidden states')
parser.add_argument('-D_dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')


## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=100,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=100,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options
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
parser.add_argument('-optim', default='adam',
                    help="Optimization method. [sgd|adagrad|adadelta|adam|rmsprop]")
parser.add_argument('-learning_rate', type=float, default=2e-5,
                    help="""Starting learning rate. If adagrad/adadelta/adam/rmsprop is
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
parser.add_argument('-start_decay_at', default=8,
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
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
parser.add_argument('-seed', type=int, default=3435,
                    help="Seed for random initialization")

opt = parser.parse_args()
opt.cuda = len(opt.gpus)

print(opt)

torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    if not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with -cuda")
    else:
        torch.cuda.manual_seed(opt.seed)

if opt.cuda:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit


def memoryEfficientLoss(G, outputs, sources, targets, criterion, optimizerG=None, D=None, optimizerD=None, eval=False):
    # compute generations one piece at a time
    loss = 0
    errD, errG, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval).contiguous()

    batch_size = outputs.size(1)
    if opt.supervision or eval:

        # Legacy code, can be used with -geneare False option
        if not opt.generate:
            outputs_split = torch.split(outputs, opt.max_generator_batches)
            targets_split = torch.split(targets.contiguous(), opt.max_generator_batches)
            for out_t, targ_t in zip(outputs_split, targets_split):

                out_t = out_t.view(-1, out_t.size(2))
                pred_t = G.generator(out_t)
                loss_t = criterion(pred_t, targ_t.view(-1))
                loss += loss_t.data[0]
                if not eval:
                    loss_t.div(batch_size).backward()
        else:
            pred_t = outputs
            targ_t = targets.contiguous()
            loss_t = criterion(pred_t, targ_t.view(-1))
            loss += loss_t.data[0]
            if not eval:
                loss_t.div(batch_size).backward()

        grad_output = None if outputs.grad is None else outputs.grad.data

    else:
        noise_sources = one_hot(G, sources.data,
                                opt.unievrsalVocabSize)
        noise_targets = one_hot(G, targets.data,
                                opt.unievrsalVocabSize)

        if opt.cuda:
            noise_sources = noise_sources.cuda()
            noise_targets = noise_targets.cuda()

        fake = torch.cat([noise_sources, outputs], 0)
        real = torch.cat([noise_sources, noise_targets], 0)

        fake = fake.contiguous().view(fake.size()[0] / opt.batch_size, opt.batch_size, fake.size()[1])
        real = real.contiguous().view(real.size()[0] / opt.batch_size, opt.batch_size, real.size()[1])


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

            for p in D.parameters():
                print('p.grad.data: ' + str(p.grad.data))

            optimizerD.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            # if i % G_train_interval == 0:
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()

            D_fake = D(fake)
            errG = -torch.mean(D_fake)

            errG.backward()
            grad_output = None if outputs.grad is None else outputs.grad.data

            D_G_z2 = D_fake.data.mean()

        else:

            # GAN variables
            real_label = 1
            fake_label = 0

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
            grad_output = None if outputs.grad is None else outputs.grad.data

            D_G_z2 = output.data.mean()

    return loss, grad_output, errD, errG, D_x, D_G_z1, D_G_z2

def one_hot(G, input, num_input_symbols):
    one_hot_tensor = torch.FloatTensor(input.size()[1], input.size()[0], num_input_symbols)
    input = torch.transpose(input, 1, 0)
    for i in range(input.size()[0]):
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(input.size()[1], num_input_symbols)
        if opt.cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.zero_()

        if opt.ST:
            y_onehot.scatter_(1, input[i].unsqueeze(1), 1)
            # Use ST gumbel-softmax
            one_hot_tensor[i] = y_onehot
        else:
            y_onehot.scatter_(1, input[i].unsqueeze(1), num_input_symbols)
            # Use soft gumbel-softmax
            pert = G.generator.real_sampler(Variable(y_onehot))
            one_hot_tensor[i] = pert.data

    one_hot_tensor = torch.transpose(one_hot_tensor,1,0)
    return Variable(one_hot_tensor.contiguous().view(one_hot_tensor.size()[0]*one_hot_tensor.size()[1], one_hot_tensor.size()[2]))

def eval(G, criterion, data):
    total_loss = 0
    total_words = 0

    G.eval()
    for i in range(len(data)):
        batch = [x.transpose(0, 1) for x in data[i]] # must be batch first for gather/scatter in DataParallel
        outputs, dec_hidden = G(batch)  # FIXME volatile
        targets = batch[1][:, 1:]  # exclude <s> from targets
        sources = batch[0]
        loss, _, _, _, _, _, _ = memoryEfficientLoss(G, outputs,
                                                     sources, targets,
                                                    criterion,None,None,None,True)

        # (G, outputs, sources, targets, criterion, optimizerG = None, D = None, optimizerD = None, eval = False)
        # loss, _ = memoryEfficientLoss(G, outputs, sources, targets, criterion, eval=False)
        total_loss += loss
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    G.train()
    return total_loss / total_words


def trainModel(G, trainData, validData, dataset, optimizerG, D=None, optimizerD=None):
    print(G)
    G.train()
    # if optimizerG.last_ppl is None:
    #     for p in G.parameters():
    #         p.data.uniform_(-opt.param_init, opt.param_init)

    if opt.supervision:
        # define criterion of each GPU
        criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    else:
        criterion = nn.BCELoss()
    ppl_eval_criterion = NMTCriterion(dataset['dicts']['tgt'].size())

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
            batch = [x.transpose(0, 1) for x in batch] # must be batch first for gather/scatter in DataParallel

            G.zero_grad()
            outputs, dec_hidden = G(batch)
            targets = batch[1][:, 1:]  # exclude <s> from targets
            sources = batch[0]

            if opt.supervision:
                loss, gradOutput, _, _, _, _, _ = memoryEfficientLoss(G, outputs,
                                                       sources, targets,
                                                       criterion)

                outputs.backward(gradOutput)

                # print('ITERATION: ')
                # for p in G.parameters():
                #     print('p.grad.data: ' + str(p.grad.data))

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
                        if opt.estimate_temp:
                            learned_temp = G.generator.learned_temp
                        else:
                            learned_temp = G.generator.scheduled_temp
                        print("Real temp: %.4f, Generated temp: %.4f " % (G.generator.real_temp, learned_temp))

                    report_loss = report_words = 0
                    start = time.time()


            else:
                _, gradOutput, errD, errG, D_x, D_G_z1, D_G_z2 = memoryEfficientLoss(G, outputs, sources,
                                                       targets, criterion,
                                                       optimizerG, D,
                                                       optimizerD)

                outputs.backward(gradOutput)

                # print('ITERATION: ')
                # for p in G.parameters():
                #     print('p.grad.data: ' + str(p.grad.data))

                optimizerG.step()

                if i % opt.log_interval == 0 and i > 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                            % (epoch, opt.epochs, i, len(trainData),
                               errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                    if opt.use_gumbel:
                        if opt.estimate_temp:
                            learned_temp = G.generator.learned_temp
                        else:
                            learned_temp = G.generator.scheduled_temp
                        print("Real temp: %.4f, Generated temp: %.4f " % (G.generator.real_temp, learned_temp))


                # anneal tau for gumbel
                if not opt.estimate_temp and i % opt.gumbel_anneal_interval == 0 and i > 0:
                    G.generator.anneal_tau_temp()

            G.generator.iter_cnt += 1

        if opt.supervision:
            return total_loss / total_words
        else:
            return 0

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))

        #  (2) evaluate on the validation set

        valid_loss = eval(G, ppl_eval_criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)

        #  (3) maybe update the learning rate
        if opt.optim == 'sgd':
            optimizerG.updateLearningRate(valid_loss, epoch)

        #  (4) drop a checkpoint
        checkpoint = {
            'model': G,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optimizerG,
        }
        torch.save(checkpoint,
                   '%s_e%d_%.2f.pt' % (opt.save_model, epoch, valid_ppl))


def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda)

    dicts = dataset['dicts']
    opt.unievrsalVocabSize = dicts['tgt'].size()
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.train_from is None:
        encoder = onmt.Models.Encoder(opt, dicts['src'])
        decoder = onmt.Models.Decoder(opt, dicts['tgt'])
        generator = onmt.Models.GANGenerator(opt, dicts['tgt'])

        if opt.cuda > 1:
                generator = nn.DataParallel(generator, device_ids=opt.gpus)
        G = onmt.Models.NMTModel(encoder, decoder, generator)

        if opt.generate:
            G.set_generate(True)

        if opt.cuda > 1:
            G = nn.DataParallel(G, device_ids=opt.gpus)
        if opt.cuda:
            G.cuda()
        else:
            G.cpu()

        G.generator = generator

        for p in G.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optimizerG = onmt.Optim(
            G.parameters(), opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
        D=None
        optimizerD=None
        if not opt.supervision:

            # If we are in GAN aetting, build up the discriminatorZ
            D = onmt.Models.D(opt, dicts['tgt'])

            for p in D.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

            optimizerG = optim.RMSprop(G.parameters(), lr=5e-5)
            optimizerD = optim.RMSprop(D.parameters(), lr=5e-5)

            # optimizerD = onmt.Optim(
            #     G.parameters(), opt.optim, opt.learning_rate, opt.max_grad_norm,
            #     lr_decay=opt.learning_rate_decay,
            #     start_decay_at=opt.start_decay_at
            # )

            if opt.cuda:
                D.cuda()
            else:
                D.cpu()
    else:
        print('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        G = checkpoint['model']
        if opt.cuda:
            G.cuda()
        else:
            G.cpu()
        optimizerG = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch'] + 1

    nParams = sum([p.nelement() for p in G.parameters()])
    print('* number of G parameters: %d' % nParams)

    if not opt.supervision:
        nParams = sum([p.nelement() for p in D.parameters()])
        print('* number of D parameters: %d' % nParams)


    trainModel(G, trainData, validData, dataset, optimizerG, D, optimizerD)


if __name__ == "__main__":
    main()
