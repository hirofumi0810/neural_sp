## How to use warpctc_pytorch
Useful documentation from [here](https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788)

The CTC loss function computes total CTC loss on a batch of sequences.
Total loss is not the equal to the sum of the losses for individual samples.
Not clear why [*](https://discuss.pytorch.org/t/how-to-fill-the-label-tensor-for-ctc-loss/5801).


ctc_loss(logits, labels, logits_sizes, label_sizes)
### logits
------------
#### Activation before the softmax layer.
Tensor of size (seq_len, batch_size, n_alphabet+1).
Note that each sample in the batch may have a different sequence length, so the seq_len size of the tensor is maximum of all sequence lengths in the batch.
The tail of short sequences should be padded with zeros.
The [0] index of the logits is reserved for "blanks" which is why the 3rd dimension is of size n_alphabet+1.

### labels
------------
#### Ground truth labels.
A 1-D tensor composed of concatenated sequences of int labels (not one-hot vectors).
Scalars should range from 1 to n_alphabet.
0 is not used, as that is reserved for blanks.
For example, if the label sequences for two samples are [1, 2] and [4, 5, 7] then the tensor is [1, 2, 4, 5, 7].

### logits_sizes
------------
#### Sequence lengths of the logits.
A 1-D tensor of ints of length batch_size.
The ith value specifies the sequence length of the logits of the ith sample that are used in computing that sample's CTC loss.
Values in the probs tensor that extend beyond this length are ignored.

### label_sizes
------------
#### Sequence lengths of the labels.
A 1-D tensor of ints of length batch_size.
The ith value specifies the sequence length of the labels of the ith sample that are used in computing that sample's CTC loss.
The length of the labels vector should be equal to the cumulative sum of the elements in the label_sizes vector.
