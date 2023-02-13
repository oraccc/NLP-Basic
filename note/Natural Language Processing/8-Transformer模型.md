## Transformer结构



### §8.1 Transformer结构简介

**[《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762.pdf)是一篇Google提出的将Attention思想发挥到极致的论文。这篇论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN**。目前大热的BERT就是基于Transformer构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。

和Seq2Seq模型一样，Transformer模型中也采用了 Encoer-Decoder 架构。

但其结构相比于Attention更加复杂，论文中Encoder层由6个Encoder堆叠在一起，Decoder层也一样。