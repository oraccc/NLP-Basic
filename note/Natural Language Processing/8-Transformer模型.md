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
PE(pos, 2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

$$
PE(pos, 2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$
其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在**偶数位置，使用正弦编码，在奇数位置，使用余弦编码**。

最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/PE.png" width="600" />

#### Self-Attention

接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是**Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路**，我们看个例子：

The animal didn't cross the street because it was too tired

这里的 it 到底代表的是 animal 还是 street 呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把 it 和 animal 联系起来，接下来我们看下详细的处理过程。

* 首先，Self-Attention会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要**和embedding的维度一样**，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是**64**。

  <img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/self-attention1.png" width="500" />

* 计算Self-Attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的**关注程度**。这个分数值的计算方法是Query与Key做**点乘**，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2。

* 接下来，把点乘的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是**每个词对于当前位置的词的相关性大小**，当然，当前位置的词相关性肯定会会很大。

* 下一步就是把**Value和softmax得到的值进行相乘**，并相加，得到的结果即是Self-Attetion在当前节点的值。

  <img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/self-attention2.png" width="450" />

在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵 Q 与 K 相乘，乘以一个常数，做softmax操作，最后乘上 V 矩阵。

**这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为scaled dot-product attention。**

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/self-attention3.png" width="350" />

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/self-attention4.png" width="550" />

#### Multi-Headed Attention

:star: 所谓自注意力机制就是通过某种运算来直接计算得到**句子在编码过程中每个位置上的注意力权重**；然后再以权重和的形式来计算得到整个句子的隐含向量表示。

:star: 自注意力机制的缺陷就是：模型在对当前位置的信息进行编码时，**会过度的将注意力集中于自身的位置**， 因此作者提出了通过多头注意力机制来解决这一问题。

实验证明，**多头注意力机制效果优于单头注意力**。

Transformer的多头注意力看上去是借鉴了CNN中同一卷积层内**使用多个卷积核**的思想，原文中使用了 8 个 scaled dot-product attention ，在同一 multi-head attention层中，输入均为 KQV ，**同时**进行注意力的计算，彼此之前**参数不共享**，最终将结果**拼接**起来，这样可以允许模型在**不同的表示子空间里学习到相关的信息**

简而言之，就是希望每个注意力头，只关注最终输出序列中一个子空间，互相**独立**。其核心思想在于，抽取到更加丰富的**特征信息**。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/multi-headed.png" width="650" />

#### Layer Normalization

