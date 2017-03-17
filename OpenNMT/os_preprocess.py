import onmt

import argparse
import os
import torch

parser = argparse.ArgumentParser(description='preprocess.lua')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train', required=True,
                    help="Path to the training source data")
parser.add_argument('-valid', required=True,
                    help="Path to the validation source data")
parser.add_argument('-test', required=True,
                    help="Path to the test source data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-universal_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-seq_length', type=int, default=25,
                    help="Maximum sequence length")
parser.add_argument('-min_seq_length', type=int, default=3,
                    help="Minimum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-report_every', type=int, default=1000000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def initOSvocabulary(name, vocabFile):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict(None, lower=opt.lower)
        vocab.loadFile(vocabFile)

        vocab.addSpecial(onmt.OS_Constants.PAD_WORD,0)
        vocab.addSpecial(onmt.OS_Constants.UNK_WORD,25001)
        vocab.addSpecial(onmt.OS_Constants.BOS_WORD,25002)
        vocab.addSpecial(onmt.OS_Constants.EOS_WORD,25003)
        vocab.addSpecial(onmt.OS_Constants.IEOS_WORD,25004)

        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')
    else:
        print('Error: vocab file required.')

    print()
    return vocab

def makeOSdata(srcFile, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % (srcFile))
    srcF = open(srcFile)

    while True:
        lines = srcF.readline().split('|')

        if not lines or lines[0] == '':
            break

        if len(lines) == 3:

            src_t = lines[0] + ' '  + str(onmt.OS_Constants.IEOS) + ' ' +  lines[1]
            src_t = map(int, src_t.split(' '))
            src_t = torch.LongTensor(src_t)

            tgt_t = lines[2]
            tgt_t = map(int, tgt_t.split(' '))
            tgt_t = torch.LongTensor(tgt_t)

            if len(src_t) <= opt.seq_length \
                    and len(src_t) >= opt.min_seq_length \
                    and len(tgt_t) <= opt.seq_length \
                    and len(tgt_t) >= opt.min_seq_length:

                src_t = tgtDicts.convertToLabels(src_t,
                                       onmt.OS_Constants.EOS)

                src += [tgtDicts.convertToIdx(src_t,
                                          onmt.OS_Constants.UNK_WORD)]


                tgt_t = tgtDicts.convertToLabels(tgt_t,
                                               onmt.OS_Constants.EOS)

                tgt += [tgtDicts.convertToIdx(tgt_t,
                                                onmt.OS_Constants.UNK_WORD,
                                                onmt.OS_Constants.BOS_WORD,
                                                onmt.OS_Constants.EOS_WORD)]



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


def main():

    dicts = {}
    dicts['src'] = initOSvocabulary('source', opt.universal_vocab)
    dicts['tgt'] = dicts['src']

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeOSdata(opt.train, dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeOSdata(opt.valid, dicts['tgt'])

    print('Preparing testing ...')
    test = {}
    test['src'], test['tgt'] = makeOSdata(opt.test, dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + 'os-train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                 'test' : test}
    torch.save(save_data, opt.save_data + 'os-train.pt')


if __name__ == "__main__":
    main()