# Introduction

This project intends to build a Seq2Seq model consisted of only decoder without distinguishing source and target from architecture design, named **Reformer**.

# Notes
- Source/Target embedding is important when source and target are mixed in the same dimension (similar to position embedding), but degrades performance when it is not the case
- Training deep network (10+) should use normalize_before for better performance
- Input/Output process is important since the representation capacity should gradually decrease to make the final decision, thus input `add` performs similar to `cat` with half parameters (avoid the growth of representation capacity) [output should not take `max` but use all to avoid sudden decrease of representation capacity]

# Issues
- for the last layer, after two layer fnn we have a project layer, which might be problematic since there are many fnn stacked together
- each head is wider than 64?
## Homogeneity
- as layer gets deeper, information might be blur for different position, retrieving lower layer information or add position indication information to each layer might be helpful
- to address the possible harm of homogeneity, model might need to make early prediction at lower layers

# TODO
- should we multiply `math.sqrt(0.5)` after residual add?
- model underfits, some parameters are useless (too few non-linear)
- apply ReLU after attention instead of itself form a sublayer
- predict next word based on representations from all source positions instead of their maximum (information loss, GLU-style)
- when compressing the last 2D representation for sequential output, reduce dimension based on the per source position vector representation instead of per scalar feature
- between any two successive attentions, a fnn & norm might required, i.e., only one attention within one block
- stacking more encoder self-attention might be helpful since deeper encoder benefits performance
- for decoder self-attention, allowing it to access the previous higher layer representation might be helpful since it can start from a better point instead of the raw embeddings
- the output projection at the end of the sublayer has the role that assigns unique weights to features at different dimensions, it could be replaced by reassign weights when these features get into different sublayers (similar to GLU and highway, self-regularization)
- the input projection at the beginning of the ffn expands the feature size, which helps to prevent the deal units in the ReLU. By replacing ReLU with other activation functions like tanh might enable the model to not use such input projection
- ODE might suits for this model since it consumes less memory
<!---
- Bidirectional Transformer Decoder: let previous generated target words representations to consider the latest generated words
- Reformer Decoder: instead of feeding the raw source embedding into the reformer, use the refined encoder representation by feeding it only to the first layer as embedding (with/without additional position embedding) or clamping it fixed to each layer
- Encoder-Memory RNN Decoder: use the encoder as the memory cell within RNN to form the decoder, that is we use every newly generated target word to update all representations within this dynamic encoder
- Dynamic Beam Search: 1) in the copy loss case, reformer will evaluate previous generated tokens differently according to the newly generated ones, thus beam search score should not be accumulated or fixed (it provides a chance to exclude / reorder hypotheses according to its new decisions, thus able to revise the hypotheses more flexibly); 2) in the next word prediction case (predict next word probability at each target positions), reformer need a way to ensemble predictions from different positions to make the final decision
-->
