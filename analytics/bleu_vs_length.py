from matplotlib import pyplot as plt
import numpy as np
import codecs
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('bleu', nargs='+', help='bleu files')
parse.add_argument('-len', type=str, required=True, help='lengths')
parse.add_argument('-escape', type=bool, default=False, help='escape long sentences')
parse.add_argument('-b', type=int, default=5, help='#bins')
parse.add_argument('-gap', type=int, default=10, help='bin width')
args = parse.parse_args()

b = args.b
gap = args.gap

fs = []
for f in args.bleu:
    with codecs.open(f, 'r', encoding='utf-8') as f:
        fs.append([line.strip() for line in f.readlines()])

with codecs.open(args.len, 'r', encoding='utf-8') as f:
    sents = [line.strip().split() for line in f.readlines()]

for data, f in zip(fs, args.bleu):
    bins = [[] for i in range(b)]

    for i in range(len(data)):
        score = eval(data[i])
        l = len(sents[i])
        if args.escape:
            try:
                bins[l // gap].append(score)
            except IndexError:
                continue
        else:
            bins[min(l // gap, b - 1)].append(score)

    stats = [sum(bin) / len(bin) for bin in bins]
    lens = [len(bin) for bin in bins]

    print('{} BLEU: {}'.format(f, stats))
    print('{} lens: {}'.format(args.len, lens))

    plt.plot(np.arange(1, b + 1) * gap, stats, label=f)

plt.xlabel('Length')
plt.ylabel('BLEU')
plt.title('BLEU vs. Length')
plt.legend()
plt.show()
