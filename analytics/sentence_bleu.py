import argparse
import codecs
import math
import sys

parse = argparse.ArgumentParser()
parse.add_argument('-trans', required=True, help='translations')
parse.add_argument('-ref', required=True, nargs='+', help='references')
parse.add_argument('-bleu', default='NIST', choices=['NIST', 'NIST-SBP'], help='bleu type')
parse.add_argument('-output', required=True, help='output file')
args = parse.parse_args()

BLEU_TYPE = args.bleu
NGRAM = 4
ref_paths = args.ref

ref_num = len(ref_paths)

assert ref_num >=1, 'you must set reference set'


def get_ngram_pool(line, ngram):
    pool = dict()
    length = len(line)
    for n in range(ngram):
        for beg in range(length - n):
            ngram_str = ' '.join(line[beg:beg + n+1])
            if ngram_str in pool:
                pool[ngram_str] += 1
            else:
                pool[ngram_str] = 1
    return pool


class Reference:
    def __init__(self, line, ngram=4):
        self.n = ngram
        tokens = line.strip().split(' ')
        self.length = len(tokens)
        self.ngram_pool = get_ngram_pool(tokens, self.n)


class MultiReference:
    def __init__(self, ref_list):
        self.ref_num = len(ref_list)
        self.sent_num = len(ref_list[0])

        self.ngram_pool = [dict() for i in range(self.sent_num)]
        self.length = []

        for i in range(self.sent_num):
            length = []
            ngrams = self.ngram_pool[i]
            for ref_id in range(self.ref_num):
                ref = ref_list[ref_id][i]
                for k,v in ref.ngram_pool.items():
                    if k not in ngrams:
                        ngrams[k] = v
                    else:
                        if v > ngrams[k]:
                            ngrams[k] = v
                length += [ref.length]
            self.length += [length]

    # returns: ngram_pool, length_list of index-th sentence
    def __getitem__(self, index):
        assert 0 <= index < self.sent_num, 'illegal index:{} while sentence num={}'.format(index, self.sent_num)
        return self.ngram_pool[index], self.length[index]

    def __len__(self):
        return self.ref_num


def read_ref_file(ref_path):
    ref_set = []
    with codecs.open(ref_path, 'r', encoding='utf-8') as f:
        for l in f:
            ref = Reference(l)
            ref_set += [ref]
    return ref_set


ref_list = []
for r in ref_paths:
    ref_list += [read_ref_file(r)]

multi_ref = MultiReference(ref_list)


# log(bleu+1) = 1/n*(sigma(i=1 to n) log( (mi+1)/(li+1) )) + min(1-r/c,0)
def get_metrics(cand, ref, bleu_type='NIST'):
    cand = cand.strip().split(' ')
    cand_len = len(cand)
    cand_pool = get_ngram_pool(cand, NGRAM)

    ref_pool = ref[0]
    ref_len = ref[1]

    # ngram precision
    ngram_acc = [0] * NGRAM
    for n in range(NGRAM):
        for beg in range(cand_len - n):
            ngram_str = ' '.join(cand[beg:beg + n+1])
            if ngram_str in ref_pool:
                ngram_acc[n] += min(ref_pool[ngram_str], cand_pool[ngram_str])
    ngram_acc = [1.0 * (v+1)/(max(0, cand_len-i) + 1) for i, v in enumerate(ngram_acc)]
    ngram_acc = [math.log(i) for i in ngram_acc]
    log_p = sum(ngram_acc)/NGRAM

    # bp
    if bleu_type == 'NIST':
        r = min(ref_len)
    elif bleu_type == 'IBM':
        r = min([math.fabs(l-cand_len) for l in ref_len])
    else:
        print('not support bleu type:%s' % bleu_type)
        sys.exit(-1)
    log_bp = min(0, 1 - 1.0 * r / cand_len)

    log_bleu = log_p + log_bp
    return math.exp(log_bleu)


with codecs.open(args.trans, 'r', encoding='utf-8') as f, codecs.open(args.output, 'w', encoding='utf-8') as of:
    for idx, l in enumerate(f):
        trans = l.strip()
        ref = multi_ref[idx]

        sentence_bleu = get_metrics(trans, ref, BLEU_TYPE)
        print('{:.4f}'.format(sentence_bleu), file=of)
