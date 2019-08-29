from matplotlib import pyplot as plt
import numpy as np
import codecs
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('-same', nargs='+', help='translations')
parse.add_argument('-trans', type=str, required=True, help='references')
parse.add_argument('-vocab', type=str, required=True, help='vocabulary')
parse.add_argument('-b', type=int, default=5, help='#bins')
parse.add_argument('-escape', type=bool, default=False, help='escape long sentences')
args = parse.parse_args()

datas = []
for idx, s in enumerate(args.same):
    with codecs.open(s, 'r', encoding='utf-8') as f:
        datas.append([[eval(e) for e in line.strip().split()] for line in f.readlines()])

with codecs.open(args.trans, 'r', encoding='utf-8') as f:
    trans = [line.strip().split() + ['<eos>'] for line in f.readlines()]

vocab = {}
with codecs.open(args.vocab, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word, freq = line.strip().split()
        vocab[word] = eval(freq)

b = args.b
gap = len(vocab) // b

for data, f in zip(datas, args.same):
    same = [0.] * b
    all_ = [0.] * b
    for l, line in enumerate(data):
        for w, s in enumerate(line):
            try:
                freq = vocab[trans[l][w]]
            except KeyError:
                continue
            if args.escape:
                try:
                    same[freq // gap] += s
                    all_[freq // gap] += 1
                except IndexError:
                    continue
            else:
                same[min(freq // gap, b - 1)] += s
                all_[min(freq // gap, b - 1)] += 1
    acc = np.asarray(same) / np.asarray(all_)
    print('{} accuracy: {}'.format(f, acc))
    plt.plot(np.arange(1, b + 1) * gap, acc, label=f)

plt.xlabel('Frequency')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Frequency')
plt.legend()
plt.show()
