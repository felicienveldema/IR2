# IR2
IR2 Project 2018

Papers:

- Generating More Interesting Responses in Neural
Conversation Models with Distributional Constraints  - https://arxiv.org/pdf/1809.01215

- https://github.com/facebookresearch/ParlAI

-Topic Aware Neural Response Generation  - http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14563/14260

-Towards Implicit Content-Introducing for Generative Short Text Conversation Systems - http://www.aclweb.org/anthology/D17-1233         - 


The code for the implemented seq2seq model can be found in the file:
**ParlAi/parlai/agents/seq2seq_custom/seq2seq_custom.py**

The PMI model is created when initializing the model (**_init_model** function). This is only done once, after that this is skipped. The training of the two models is done in the function **train_step**

The run the training process when in ParlAI folder:

```console
sh twitter.sh
```
