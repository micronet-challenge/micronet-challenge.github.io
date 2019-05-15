# Overview
Contestants will compete to build the most efficient model that solves the target task to the specified quality level. The competition is focused on efficient inference, and uses a theoretical metric rather than measured inference speed to score entries. We hope that this encourages a mix of submissions that are useful on today’s hardware and that will also guide the direction of new hardware development.

For this iteration of the competition, the efficiency of a model will be measured by a combination of the number of math operations required to make predictions and the number of bytes required to store the model parameters. Our goal is for our scoring metric to be indicative of performance that could be achieved in hardware. However, we do not want to unnecessarily burden competitors with metrics that are complex or difficult to reason about. For this first iteration of the competition, we’ve decided to favor simplicity, and we’ve included details on other metrics we’d like to take into account for future iterations of this competition in the appendix of the [scoring page](./scoring_and_submission).

In addition to incentivizing the development of efficient models and model compression techniques, we hope this competition provides a forum for more rigorous benchmarking and comparison of existing techniques and for the study of combinations of approaches like sparsity, quantization, distillation, and neural architecture search.

# Motivation & Philosophy
The broader goal of this competition is to incentivize the co-design of neural networks architectures and hardware accelerators. We’re interested in understanding how optimizations and tradeoffs can be made at different levels of the stack to build a more efficient end-to-end system.

An ideal competition for this might entail contestants submitting both neural network architectures and hardware designs on which their architecture would run. However, these skill sets rarely overlap and the hardware design cycle is very long relative to the time it takes to develop neural network architectures. Instead, we plan to measure a suite of theoretical properties of neural network architectures, and to combine them in a way that accurately approximates performance that could be achieved with specialized hardware. Over time, we plan to grow this set of metrics and to refine our scoring system to more accurately capture hardware design constraints.

We hope that this competition will help guide future hardware designs, spur the development of efficient software libraries on existing hardware, and stimulate progress in the development of efficient neural network architectures.

# Tasks
The competition consists of three different tasks. Contestants are free to submit entries for one, two, or all three tasks. Contestants are allowed to enter up to three models for each task, but will be ranked according to their top entry in each task. Entries can only be trained on the training data for the task they are entered in. No pre-training, or use of auxiliary data is allowed.

Contestants may work together in teams, but each individual is only allowed three entries per task across all the teams they participate with. If an entry is submitted by a team of contestants, and any member of that team has already contributed to three entries, the new entry will be disqualified. The three tasks are:

[_ImageNet Classification:_](http://image-net.org/index) The de facto standard dataset for image classification. The dataset is composed of 1,281,167 training images and 50,000 development images. Entries are required to achieve 75% top-1 accuracy on the public test set.

[_CIFAR-100 Classification:_](https://www.cs.toronto.edu/~kriz/cifar.html) A widely popular image classification dataset of small images. The dataset is composed of 50,000 training images and 10,000 development images. Entries are required to achieve 80% top-1 accuracy on the test set.

[_WikiText-103 Language Modeling:_](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) A language modeling dataset that emphasizes long-term dependencies. Entries will perform the standard language modeling task, predicting the next token from the current one. The dataset is composed of 103 million training words, 217 thousand development words, and 245 thousand testing words. Entries should use the standard word-level vocabulary of 267,735 tokens. Entries are required to achieve a word-level perplexity below 35 on the test set.

# Important Dates
__Submission Deadline:__ Midnight Pacific Time, September 30th, 2019.

# Contact Us
Feel free to reach contact us at <micronet.challenge@gmail.com> with any questions you might have.

# Organizers
Trevor Gale - Google Brain  
Erich Elsen - DeepMind  
Olivier Temam - DeepMind  
Scott Gray - OpenAI  
Jongsoo Park - Facebook  
Cliff Young - Google Brain  
Sara Hooker - Google Brain  
Niki Parmar - Google Brain  
Ashish Vaswani - Google Brain