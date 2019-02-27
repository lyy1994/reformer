# Introduction

This project intends to build a Seq2Seq model consisted of only decoder without distinguishing source and target from architecture design, named **Reformer**.

# Notes
- Source/Target embedding is important when source and target are mixed in the same dimension (similar to position embedding), but degrades performance when it is not the case
- Training deep network (10+) should use normalize_before for better performance
- Input/Output process is important since the representation capacity should gradually decrease to make the final decision, thus input `add` performs similar to `cat` with half parameters (avoid the growth of representation capacity) [output should not take `max` but use all to avoid sudden decrease of representation capacity]
- A clean residual connection requires each block to end with a linear transformation since it serves as the weighting scheme to the current layer's features when representations from different layers are summed

# Issues
- Representation size scaled `O(n)` to the source input with size `n` brings large memory consumption as well as the overfitting risk 
