Contestants will compete to build the most efficient model that solves the target task to the specified quality level. The competition is focused on efficient inference, and uses a theoretical metric rather than measured inference speed to score entries. We hope that this encourages a mix of submissions that are useful on today’s hardware and that will also guide the direction of new hardware development.

For this iteration of the competition, the efficiency of a model will be measured by a combination of the number of math operations required to make predictions and the number of bytes required to store the model parameters. Our goal is for our scoring metric to be indicative of performance that could be achieved in hardware. However, we do not want to unnecessarily burden competitors with metrics that are complex or difficult to reason about. For this first iteration of the competition, we’ve decided to favor simplicity, and we’ve included details on other metrics we’d like to take into account for future iterations of this competition in the appendix.

In addition to incentivizing the development of efficient models and model compression techniques, we hope this competition provides a forum for more rigorous benchmarking and comparison of existing techniques and for the study of combinations of approaches like sparsity, quantization, distillation, and neural architecture search.

# Contact Us
Feel free to reach contact us at `micronet.challenge@gmail.com` with any questions you might have.

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