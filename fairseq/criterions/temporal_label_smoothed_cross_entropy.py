# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('temporal_label_smoothed_cross_entropy')
class TemporalLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, sample_size = self.compute_loss(model, net_output, sample, reduce=reduce)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            # ntokens -> average nll_loss
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            # sample_size -> average loss
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        sample_size = 0
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        next_mask = model.get_next_mask(sample, net_output).view(-1, 1)
        # since we compute loss per target time step, we take future
        # mask into account
        raw_nll_loss = -lprobs.gather(dim=-1, index=target)
        nll_loss = raw_nll_loss[non_pad_mask * next_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask * next_mask]
        sample_size += (non_pad_mask * next_mask).sum()
        # since we compute loss per target time step and nll_loss will be
        # used to compute ppl, we retain only the next step prediction loss
        ppl_loss = raw_nll_loss[non_pad_mask * next_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            ppl_loss = ppl_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        # we multiple the target sentence length to avoid averaging ntokens over time
        ppl_loss *= sample['target'].size(1)
        return loss, ppl_loss, sample_size.item()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
