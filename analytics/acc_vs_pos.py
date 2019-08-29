from matplotlib import pyplot as plt
import numpy as np
import codecs
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('-same', nargs='+', help='translations')
parse.add_argument('-b', type=int, default=5, help='#bins')
parse.add_argument('-gap', type=int, default=10, help='bin width')
parse.add_argument('-escape', type=bool, default=False, help='escape long sentences')
args = parse.parse_args()

b = args.b
gap = args.gap

datas = []
maxlens = [0] * len(args.same)
for idx, f in enumerate(args.same):
    with codecs.open(f, 'r', encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            es = line.strip().split()
            maxlens[idx] = max(maxlens[idx], len(es))
            data.append([eval(e) for e in es])
        datas.append(data)

for idx, (data, f) in enumerate(zip(datas, args.same)):
    s, a = 0., 0.
    for l in data:
        s += sum(l)
        a += len(l)
    print('{} (overall) accuracy: {:.4f}'.format(f, s / a))

    same = [0.] * b
    all_ = [0.] * b
    for l in data:
        for i, e in enumerate(l):
            if args.escape:
                try:
                    same[i // gap] += e
                    all_[i // gap] += 1
                except IndexError:
                    continue
            else:
                same[min(i // gap, b - 1)] += e
                all_[min(i // gap, b - 1)] += 1
    acc = np.asarray(same) / np.asarray(all_)
    print('{} (per pos) accuracy: {}'.format(f, acc))
    plt.plot(np.arange(1, b + 1) * gap, acc, label=f)

plt.xlabel('Position')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Position')
plt.legend()
plt.show()
