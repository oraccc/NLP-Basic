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

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/encoders.png" width="600" />

#### Positional Encoding

Transformer模型中缺少一种解释输入序列中单词顺序的方法，它跟序列模型还不一样。为了处理这个问题，Transformer给 Encoder 层和 Decoder 层的输入添加了一个额外的向量Positional Encoding，**维度和embedding的维度一样**，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。

这个位置向量的具体计算方法有很多种，论文中的计算方法如下：
$$
PE(pos, 2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE(pos, 2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$
其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在**偶数位置，使用正弦编码，在奇数位置，使用余弦编码**。

最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/PE.png" width="600" />

#### Self-Attention

接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是**Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路**，我们看个例子：

The animal didn't cross the street because it was too tired

这里的 it 到底代表的是 animal 还是 street 呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把 it 和 animal 联系起来，接下来我们看下详细的处理过程。

* 首先，Self-Attention会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是64。