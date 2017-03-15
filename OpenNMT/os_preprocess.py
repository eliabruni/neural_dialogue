import onmt

import argparse
import os
import torch

parser = argparse.ArgumentParser(description='preprocess.lua')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")
parser.add_argument('-os_data', type=bool, default=False,
                    help='Whether to use opensubtitles dataset')

parser.add_argument('-train', required=True,
                    help="Path to the training source data")
parser.add_argument('-valid', required=True,
                    help="Path to the validation source data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-universal_vocab_size', type=int, default=25000,
                    help="Size of the source vocabulary")
parser.add_argument('-src_vocab_size', type=int, default=24999,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=24996,
                    help="Size of the target vocabulary")
parser.add_argument('-universal_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-seq_length', type=int, default=25,
                    help="Maximum sequence length")
parser.add_argument('-min_seq_length', type=int, default=3,
                    help="Minimum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD])

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + vocab.size() + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def initOSvocabulary(name, vocabFile):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + vocab.size() + ' ' + name + ' words')
    else:
        print('Error: vocab file required.')

    print()
    return vocab

def makeOSdata(srcFile):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % (srcFile))
    srcF = open(srcFile)

    while True:
        lines = srcF.readline().split('|')

        src_t = lines[0] + ' '  + onmt.Constants.IEOS + ' ' +  lines[1]
        src_t = map(int, src_t.split(' '))
        src_t = torch.LongTensor(src_t)

        tgt_t = lines[2]
        tgt_t = map(int, tgt_t.split(' '))
        tgt_t = torch.LongTensor(tgt_t)

        if len(src_t) <= opt.seq_length \
                and len(src_t) >= opt.min_seq_length \
                and len(tgt_t) <= opt.seq_length \
                and len(tgt_t) >= opt.min_seq_length:
            src += src_t
            tgt += tgt_t

            sizes += [len(src_t)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        srcWords = srcF.readline().split()
        tgtWords = tgtF.readline().split()

        if not srcWords or not tgtWords:
            if srcWords and not tgtWords or not srcWords and tgtWords:
                print('WARNING: source and target do not have the same number of sentences')
            break

        if len(srcWords) <= opt.seq_length \
                and len(srcWords) >= opt.min_seq_length \
                and len(tgtWords) <= opt.seq_length \
                and len(tgtWords) >= opt.min_seq_length:

            src += [srcDicts.convertToIdx(srcWords,
                                          onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt


def main():

    dicts = {}
    dicts['src'] = initOSvocabulary('source', opt.universal_vocab)
    dicts['tgt'] = dicts['src']

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeOSdata(opt.train)

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeOSdata(opt.valid)

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '-train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '-train.pt')


if __name__ == "__main__":
    main()