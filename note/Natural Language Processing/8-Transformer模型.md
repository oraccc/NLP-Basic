## Transformer结构

- [Transformer结构](#transformer结构)
  - [§8.1 Transformer结构简介](#81-transformer结构简介)
  - [§8.2 Encoder层](#82-encoder层)
    - [Positional Encoding](#positional-encoding)
    - [Self-Attention](#self-attention)
    - [Multi-Headed Attention](#multi-headed-attention)
    - [Layer Normalization](#layer-normalization)
      - [Batch Normalization](#batch-normalization)
      - [Layer Normalization](#layer-normalization-1)
    - [Feed Forward Neural Network](#feed-forward-neural-network)
  - [§8.3 Decoder层](#83-decoder层)
    - [Masked Mutil-Head Attention](#masked-mutil-head-attention)
    - [Cross Attention](#cross-attention)
    - [Output](#output)
  - [§8.4 与其余模型比较](#84-与其余模型比较)
    - [相比于RNN/LSTM](#相比于rnnlstm)
    - [相比于Seq2Seq](#相比于seq2seq)


### §8.1 Transformer结构简介

> :memo: [Transformer代码](https://github.com/Kyubyong/transformer)
>
> :memo: [部分代码解释](https://github.com/oraccc/NLP-Basic/blob/master/code/Natural%20Language%20Processing/8-transformer.ipynb)

**[《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762.pdf)是一篇Google提出的将Attention思想发挥到极致的论文。这篇论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN**。目前大热的BERT就是基于Transformer构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。

和Seq2Seq模型一样，Transformer模型中也采用了 `Encoer-Decoder` 架构。

但其结构相比于Attention更加复杂，论文中Encoder层由6个Encoder堆叠在一起，Decoder层也一样。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/transformer-structure.png" width="650" />

每一个 Encoder 和 Decoder 的内部结构如下图

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/encoder-decoder.png" width="450" />

- **Encoder** 包含两层，一个`Self-Attention`层和一个前馈神经网络，`Self-Attention`能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。
- **Decoder** 与Encoder的区别在于，有两个`MultiHead Attention`：
  - 底层是 **Masked MultiHead Attention**
  - 中间的MultiHead Attention可以叫做 **Cross Attention**，正是这个组件将 Encoder 和 Decoder 连接起来


---



### §8.2 Encoder层

首先，模型需要对输入的数据进行一个**embedding**操作，也可以理解为类似`w2v`的操作，embedding结束之后，输入到encoder层，`self-attention`处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/encoders.png" width="600" />

#### Positional Encoding

Transformer模型中缺少一种解释输入序列中单词顺序的方法，它跟序列模型还不一样。为了处理这个问题，Transformer给 Encoder 层和 Decoder 层的输入添加了一个额外的向量`Positional Encoding`，**维度和`Embedding`的维度一样**，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。

这个位置向量的具体计算方法有很多种，论文中的计算方法如下：
$$
PE(pos, 2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

$$
PE(pos, 2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$
其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在**偶数位置，使用正弦编码，在奇数位置，使用余弦编码**。

最后把这个`Positional Encoding`与`Embedding`的值相加，作为输入送到下一层。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/PE.png" width="600" />

#### Self-Attention

接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是**Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路**，我们看个例子：

> The animal didn't cross the street because it was too tired

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



:question: 为什么在进行softmax之前进行**scaled**（为什么除以dk的平方根）？

> 向量的点积结果会很大，将softmax函数push到梯度很小的区域，scaled会缓解这种现象。
>
> 为什么使用维度的根号？
>
> 方差越大也就说明，点积的数量级越大（以越大的概率取大值）
>
> $D(\frac{q \cdot k}{\sqrt{d_k}}) = \frac{d_k}{(\sqrt{d_k})^2}=1$
>
> 将方差控制为1，也就有效地控制了前面提到的梯度消失的问题

#### Multi-Headed Attention

:star: 所谓自注意力机制就是通过某种运算来直接计算得到**句子在编码过程中每个位置上的注意力权重**；然后再以权重和的形式来计算得到整个句子的隐含向量表示。

:star: 自注意力机制的缺陷就是：模型在对当前位置的信息进行编码时，**会过度的将注意力集中于自身的位置**， 因此作者提出了通过多头注意力机制来解决这一问题。

实验证明，**多头注意力机制效果优于单头注意力**。

:question: 为什么需要进行**Multi-Head Attention**?

> Transformer的多头注意力看上去是借鉴了CNN中同一卷积层内**使用多个卷积核**的思想，原文中使用了 8 个 scaled dot-product attention ，在同一 multi-head attention层中，输入均为 KQV ，**同时**进行注意力的计算，彼此之前**参数不共享**，最终将结果**拼接**起来，这样可以允许模型在**不同的表示子空间里学习到相关的信息**
>
> 简而言之，就是希望每个注意力头，只关注最终输出序列中一个子空间，互相**独立**。其核心思想在于，抽取到更加丰富的**特征信息**。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/multi-headed.png" width="650" />

#### Layer Normalization

在Transformer中，每一个子层（`Self-Attetion`，`Feed Forward Neural Network`）之后都会接一个**Add & Norm**，其中 Add 为**Residule Block 残差模块**，并且 Norm 有一个**Layer Normalization**。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/layer-normalization.png" width="450" />

Normalization有很多种，但是它们都有一个共同的目的，那就是把输入转化成均值为0方差为1的数据。我们在把数据送入激活函数之前进行normalization（归一化），因为我们不希望输入数据落在激活函数的饱和区。

##### Batch Normalization

BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。**BN的具体做法就是对每一小批数据，在批这个方向上做归一化**。

* 对每一个batch进行操作，使得对于这一个batch中所有的输入数据，它们的每一个特征都是均值为0，方差为1的分布 

* 单纯把所有的输入限制为(0,1)分布也是不合理的，这样会降低数据的表达能力（第L层辛苦学到的东西，这里都暴力变成(0,1)分布了）。因此需要再加一个线性变换操作，让数据恢复其表达能力。这个线性变化中的两个参数 $\gamma$, $\beta$ 是需要模型去学习的。

  

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/BN.jpg" width="750" />

##### Layer Normalization

整体做法类似于BN，不同的是LN不是在特征间进行标准化操作（横向操作），而是在整条数据间进行标准化操作**（纵向操作）**。它也是归一化数据的一种方式，不过**LN 是在每一个样本上计算均值和方差**，而不是BN那种在批方向计算均值和方差！公式如下:
$$
LN(x_i) = \alpha * \frac{x_i-\mu _L}{\sqrt{\sigma^2_L + \varepsilon}} + \beta
$$


<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/LN.jpg" width="750" />

两种方式的比较：

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/normalization.png" width="650" />

#### Feed Forward Neural Network

我们需要一种方式，把 8 个矩阵降为 1 个，首先，我们把 8 个矩阵连在一起，这样会得到一个大的矩阵，再随机初始化一个矩阵和这个组合好的矩阵相乘，最后得到一个最终的矩阵。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/ffnn.png" width="650" />

---



### §8.3 Decoder层

Decoder部分其实和 Encoder 部分大同小异，刚开始也是先添加一个位置向量Positional Encoding，接下来接的是**Masked Mutil-head Attention** 与 **Cross Attention**

#### Masked Mutil-Head Attention

**Mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果**。Transformer 模型里面涉及两种 mask，分别是 `padding mask` 和 `sequence mask`。其中，padding mask 在**所有的** `scaled dot-product attention` 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。

* **Padding Mask**

  * 因为每个批次输入序列长度是不一样的，因此我们要对输入序列进行对齐。
    * 具体来说，就是给在较短的序列后面**填充 0**。但是如果输入的序列太长，则是**截取**左边的内容，把多余的直接舍弃。
  * 因为这些填充的位置，其实是没什么意义的，所以我们的attention机制**不应该把注意力放在这些位置上**，所以我们需要进行一些处理。
  * 具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会**接近0**！而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 False 的地方就是我们要进行处理的地方。
  
* **Sequence Mask**

  * Sequence Mask 是为了使得 Decoder **不能看见未来的信息**。也就是对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该**只能依赖于 t 时刻之前的输出**，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。

    <img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/masked-attention.png" width="450" />

  * 那么具体怎么做呢？**产生一个下三角矩阵，下三角上半部分的值全为0**。把这个矩阵作用在每一个序列上，就可以达到我们的目的。

  * 对于 decoder 的 self-attention，里面使用到的 `scaled dot-product attention`，同时需要`padding mask` 和 `sequence mask` 作为 attn_mask，具体实现就是**两个mask相加**作为attn_mask。

  * 其他情况，attn_mask 一律等于 padding mask

#### Cross Attention

Decoder模块中间的部分即Cross Attention， 主要的区别在于其中 Self-Attention 的 K, V矩阵不是使用上一个 Decoder block 的输出计算的，而是**使用 Encoder 的的最终输出来计算的**。

根据 Encoder 的输出计算得到 K, V，根据上一个 Decoder block 的输出 Z 计算 Q，这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 Mask)。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/cross-attention.png" width="550" />

#### Output

当decoder层全部执行完毕后，只需要在结尾再添加一个全连接层和softmax层，假如我们的词典是1w个词，那最终softmax会输入1w个词的概率，概率值最大的对应的词就是我们最终的结果。

---



### §8.4 与其余模型比较

#### 相比于RNN/LSTM

* RNN系列的模型并行计算能力很差。
  * 因为 T 时刻的计算依赖 T-1 时刻的隐层计算结果，而 T-1 时刻的计算依赖 T-2 时刻的隐层计算结果，如此下去就形成了所谓的**序列依赖关系**。

* Transformer的特征抽取能力比RNN系列的模型要好。

#### 相比于Seq2Seq

* Seq2Seq最大的问题在于**将Encoder端的所有信息压缩到一个固定长度的向量中**，并将其作为Decoder端首个隐藏状态的输入，来预测Decoder端第一个单词(token)的隐藏状态。在输入序列比较长的时候，这样做显然会损失**Encoder端的很多信息**，而且这样一股脑的把该固定向量送入Decoder端，Decoder端不能够关注到其想要关注的信息。

* Transformer不但对Seq2Seq模型这两点缺点有了实质性的改进(多头交互式attention模块)，而且还引入了self-attention模块，让源序列和目标序列首先“自关联”起来，这样的话，源序列和目标序列自身的embedding表示所蕴含的信息更加丰富，而且后续的FFN层也增强了模型的表达能力，并且Transformer并行计算的能力是远远超过seq2seq系列的模型

---

