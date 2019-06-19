# Scoring

Two factors will be taken into account when scoring an entry:

__1. Parameter Storage:__

_Definition & Counting:_ The number of parameters that are required to perform inference. In addition to trainable parameters, any values that are needed to perform inference should be counted (e.g., indices for sparse matrix formats). Entries are allowed to assume that their models are quantized to 16-bits at no accuracy penalty. That is to say, entrants can calculate parameter storage for their models as if it were quantized to 16-bits without actually doing the quantization. A 16-bit parameter counts as one parameter. If quantization is performed, a parameter of less than 16-bits will be counted as a fraction of one parameter. For example, an 8-bit parameter counts as ½ a parameter towards the model’s parameter storage requirements.

_Rationale:_ When designing hardware, the number of bits of parameters in a model dictates the number of values that must be moved to and from arithmetic units, how much fast, on-chip memory would be needed to store the model, and the power consumption and chip area requirements for multiplication and addition circuits. 

Quantization can significantly reduce parameter storage and math requirements, but can be non-trivial to perform in popular deep learning frameworks. Our goal with the system explained above is to balance between rewarding quantization approaches while mitigating the complexity of entering the competition. Without the above rule, it’s likely that all entries would be required to perform some kind of quantization to be competitive.

Parameter efficiency is also an important metric from a modeling perspective, as it captures a models ability use it’s weights effectively.

__2. Math Operations:__

_Definition & Counting:_ The mean number of arithmetic operations per example required to perform inference on the test set. Multiplies and additions count separately. Transcendental function evaluations and bitwise operations count as one op. Dynamic activation sparsity (e.g., from ReLU activation functions) should not be taken into account.

Entries are allowed to assume that their models are quantized to 16-bits at no accuracy penalty. That is to say, entrants can calculate math operations for their models as if it were quantized to 16-bits without actually doing the quantization. A 16-bit operation counts as one operation. If quantization is performed, an operation on data of less than 16-bits will be counted as a fraction of one operation, where the numerator is the maximum number of bits in the inputs of the operation and the denominator is 16. For example, a multiplication operation with one 3-bit and one 5-bit input, with a 7-bit output, will count as 5/16th of an operation. A multiplication operation where one input is 16-bits, the other input is 8-bits and the output is 8-bits will count as one whole op, as the first input is 16-bits.

For WikiText-103, numbers can be calculated on the tokenized version of the test set, and will be averaged per-token. Numbers with WikiText-103 should be calculated with a batch size of 1 sequence, such that no padding is needed to handle variable sequence lengths. Numbers with WikiText-103 should also be calculated as if inference is performed on-line, i.e. tokens are fed sequentially to the model and it must predict the next token prior to receiving it. This is important for architectures that do not have any state (e.g., Transformer).

_Rationale:_ The number and resolution (bit-width of operands) of mathematical operations dictates how much work has to be done at inference time. Reduced precision can be exploited to save power and chip area.

We count transcendental functions as a single operation under the assumption that they can be accelerated by hardware if necessary.

We do not take dynamic activation sparsity into account because we do not take activation storage size into account. Thus, we cannot incorporate the storage and bandwidth cost of storing activations as sparse tensors into a submission’s score. In future iterations of the competition, we plan to include a metric for maximum transient activation size (see appendix) and will allow entries to take dynamic activation sparsity into account.

We allow static activation sparsity (e.g., [sparse attention](https://openai.com/blog/sparse-transformer/)) because the sparsity pattern is known ahead of time and we can avoid expensive dynamic compression of layer outputs and potentially mitigate load-balancing issues.

Note that this rule against dynamic activation sparsity is not intended to rule out [mixture-of-experts](https://arxiv.org/abs/1701.06538) (MoE) approaches, which could be viewed as a form of dynamic activation sparsity. If you’re interested in entering the competition with MoE model, reach out to the organizers at <micronet.challenge@gmail.com>.

# Ranking Entries

Entries that finish in the top 10% of one of these metrics will earn the distinction of a “highly storage-efficient solution” or “highly compute-efficient solution” for parameter storage and math operations respectively. To select overall winners for each task, we will rank entries based on the sum of these two values, with each value normalized by a baseline state-of-the-art model for the task. The entry with the lowest overall score for each task will be declared the winner.

For ImageNet and CIFAR-100, parameter storage, and compute requirements will be normalized relative to [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf) with width 1.4 (6.9M parameters, 1170M math operations). For example, an ImageNet submission that has 3 million parameters and 500 million math operations will have a final score of 3M / 6.9M + 500M / 1170M = .862.

For WikiText-103, these metrics will be normalized relative to the LSTM model presented [here](https://arxiv.org/pdf/1803.10049.pdf). This model uses a single-layer LSTM with 2048 hidden units, tied embedding and softmax parameters with dimension 512, and a linear projection to the embedding/softmax width of 512 from the LSTM width of 2048. In total, it has 159M parameters, and 318M math operations. 

# Submitting

Participants must submit their code, final checkpoint, and a writeup on their approach through a private GitHub repository. __The repository must be made public after the submission deadline passes__. The writeup must include details on the approach as well as documentation on how to reproduce the results using the provided code. Test set performance with the provided checkpoint will be verified to ensure reproducibility. __Test set performance should be easily verifiable with the provided code and checkpoint__. When submitting an entry, contestants are required to calculate the two scoring metrics for their entries and __thoroughly document their calculation__. Calculations for top entries will be manually verified by the organizers.

To submit, add [micronet-challenge-submissions](https://github.com/micronet-challenge-submissions) as a collaborator on your GitHub repository. Please see the full eligibility and participation terms [here](./micronet_global_terms.pdf).

# Appendix

_Maximum Transient Activation Size:_ When performing inference, our ability to keep activations and weights in fast, on-chip memory can be a key factor in the runtime of our model. While the number of parameters that must be stored is captured by our above metrics, we do not currently take activation storage requirements into account. 

Our ability to keep activations on chip is governed by the maximum amount of activations that must be kept around at any point during inference, which we refer to as maximum transient activation size (MTAS). Depending on the model, and batching choices, transient activations storage can exceed parameters storage, and thus have as much, if not more, impact than parameters.

While we would like to include this as a metric for this competition, it is difficult to calculate the exact maximum transient activation size on complex computational graphs with large numbers of branches. While we explored approximations, we decided that we should favor simplicity in our metrics for the first iteration of the competition. In future iterations of this competition, we would like to include MTAS, and to provide tooling to calculate it automatically for models written in popular frameworks.
