## Transformer结构



### §8.1 Transformer结构简介

**[《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762.pdf)是一篇Google提出的将Attention思想发挥到极致的论文。这篇论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN**。目前大热的BERT就是基于Transformer构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。

和Seq2Seq模型一样，Transformer模型中也采用了 Encoer-Decoder 架构。

但其结构相比于Attention更加复杂，论文中Encoder层由6个Encoder堆叠在一起，Decoder层也一样。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/transformer-structure.png" width="650" />

每一个encoder和decoder的内部结构如下图

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/encoder-decoder.png" width="450" />

- **Encoder** 包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。
- **Decoder** 也包含Encoder提到的两层网络，但是在这两层中间还有一层attention层，帮助当前节点获取到当前需要关注的重点内容。

---



### §8.2 Encoder层

首先，模型需要对输入的数据进行一个**embedding**操作，也可以理解为类似w2v的操作，embedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。