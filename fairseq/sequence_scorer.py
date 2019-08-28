# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, models, tgt_dict, top_k):
        self.models = models
        self.pad = tgt_dict.pad()
        self.top_k = top_k

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def score_batched_itr(self, data_itr, cuda=False, timer=None):
        """Iterate over a batched dataset and yield scored translations."""
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if timer is not None:
                timer.start()
            pos_scores, attn, same = self.score(s)
            for i, id in enumerate(s['id'].data):
                # remove padding from ref
                src = utils.strip_pad(s['net_input']['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                tgt_len = ref.numel()
                pos_scores_i = pos_scores[i][:tgt_len]
                same_i = same[i][:tgt_len]
                score_i = pos_scores_i.sum() / tgt_len
                if attn is not None:
                    attn_i = attn[i]
                    _, alignment = attn_i.max(dim=0)
                else:
                    attn_i = alignment = None
                hypos = [{
                    'tokens': ref,
                    'score': score_i,
                    'attention': attn_i,
                    'alignment': alignment,
                    'positional_scores': pos_scores_i,
                    'same': same_i,
                }]
                if timer is not None:
                    timer.stop(s['ntokens'])
                # return results in the same format as SequenceGenerator
                yield id, src, ref, hypos

    def score(self, sample):
        """Score a batch of translations."""
        net_input = sample['net_input']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in self.models:
            with torch.no_grad():
                model.eval()
                decoder_out = model.forward(**net_input)
                attn = decoder_out[1]
                if type(attn) is dict:
                    attn = attn['attn']

            probs = model.get_normalized_probs(decoder_out, log_probs=False, sample=sample).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        same = self.in_topk(avg_probs, sample['target'], self.top_k, self.pad)
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        avg_probs = avg_probs.gather(
            dim=2,
            index=sample['target'].data.unsqueeze(-1),
        )
        return avg_probs.squeeze(2), avg_attn, same

    @staticmethod
    def in_topk(predictions, labels, k, pad):
        """
        Returns a 0-1 tensor with the same shape as labels, 0 implies not in top-k and 1 otherwise
        :param predictions: B x T x V
        :param labels: B x T
        :param k: int
        :param pad: <pad> index
        :return: B x T
        """
        effective_k = min(k, predictions.size(-1))
        _, indices = torch.topk(predictions, effective_k)  # B x T x k
        padding_mask = torch.ne(labels, pad)  # B x T
        labels = labels.unsqueeze(-1).expand_as(indices)  # B x T x k
        same = torch.sum(torch.eq(labels, indices), -1)  # B x T
        same *= padding_mask.long()
        return same
