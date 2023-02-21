## BERT模型



### §9.1 BERT模型简介

**BERT的全称是Bidirectional Encoder Representation from Transformers**，是Google2018年提出的**预训练**模型，即双向 `Transformer` 的 `Encoder`，因为 `Decoder` 是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了 **Masked LM** 和 **Next Sentence Prediction** 两种方法分别捕捉词语和句子级别的representation。

---



### §9.2 从Word Embedding 到 BERT 模型的发展

#### :one:图像的预训练

自从深度学习火起来后，预训练过程就是做图像或者视频领域的一种比较常规的做法，有比较长的历史了，而且这种做法很有效，能明显促进应用的效果。

##### 图像预训练的步骤

- 我们设计好网络结构以后，对于图像来说一般是CNN的多层叠加网络结构，可以先用某个训练集合比如训练集合A或者训练集合B对这个网络进行预先训练，在A任务上或者B任务上学会网络参数，然后存起来以备后用。
- 假设我们面临第三个任务C，网络结构采取相同的网络结构，在比较浅的几层CNN结构，网络参数初始化的时候可以加载A任务或者B任务学习好的参数，其它CNN高层参数仍然随机初始化。
- 之后我们用C任务的训练数据来训练网络，此时有两种做法：
  - **一种**是浅层加载的参数在训练C任务过程中不动，这种方法被称为`Frozen`;
  - **另一种**是底层网络参数尽管被初始化了，在C任务训练过程中仍然随着训练的进程不断改变，这种一般叫`Fine-Tuning`，顾名思义，就是更好地把参数进行调整使得更适应当前的C任务。

##### 优点

* 如果手头任务C的训练集合数据量较少的话，利用预训练出来的参数来训练任务C，加个预训练过程也能**极大加快任务训练的收敛速度**，所以这种预训练方式是老少皆宜的解决方案，另外疗效又好，所以在做图像处理领域很快就流行开来。

##### :star:为什么预训练可行

* 对于层级的CNN结构来说，不同层级的神经元学习到了不同类型的图像特征，**由底向上特征形成层级结构**

* 预训练好的网络参数，尤其是**底层的网络参数抽取出特征跟具体任务越无关，越具备任务的通用性**，所以这是为何一般用底层预训练好的参数初始化新任务网络参数的原因。

* 而高层特征跟任务关联较大，实际可以不用使用，或者采用 `Fine-tuning` 用新数据集合清洗掉高层无关的特征抽取器。



#### :two:Word Embedding

[NNLM](https://github.com/oraccc/NLP-Basic/blob/master/note/Natural%20Language%20Processing/1-%E8%AF%8D%E5%B5%8C%E5%85%A5.md#one-nnlmneural-network-language-model) 的学习任务是根据输入某个句中单词 $W_t = BERT$ 前面句子的 $t-1$个单词，要求网络正确预测单词 `BERT`，即最大化 $P(W_t  = BERT|W_1,W2,...,W_{t-1};\theta)$

* 前面任意单词 $W_i$ 用 `Onehot` 编码（比如：0001000）作为原始单词输入，之后乘以矩阵Q后获得向量$C(W_i)$，每个单词的$C(W_i)$拼接上 `hidden layer`，然后接softmax去预测后面应该后续接哪个单词。其中 $C(W_i)$ 其实就是单词对应的 `Word Embedding` 值，矩阵Q包含V行，V代表词典大小，每一行内容代表对应单词的Word embedding值。

* Q的内容也是网络参数，需要学习获得，训练刚开始用随机值初始化矩阵Q，当这个网络训练好之后，矩阵Q的内容被正确赋值，每一行代表一个单词对应的Word embedding值。通过训练得到中间产物--**词向量矩阵Q**，这就是我们要得到的文本表示向量矩阵。

2013年最火的用语言模型做Word Embedding的工具是Word2Vec，后来又出了Glove，这些模型做法就是18年之前NLP领域里面采用预训练的典型做法。相关笔记如下

> [Word2Vec](https://github.com/oraccc/NLP-Basic/blob/master/note/Natural%20Language%20Processing/1-%E8%AF%8D%E5%B5%8C%E5%85%A5.md)
>
> [GloVe](https://github.com/oraccc/NLP-Basic/blob/master/note/Natural%20Language%20Processing/3-%E5%85%A8%E5%B1%80%E5%90%91%E9%87%8F%E8%AF%8D%E5%B5%8C%E5%85%A5.md)



#### :three: ELMO

ELMO是 `“Embedding from Language Models”` 的简称，其实这个名字并没有反应它的本质思想，提出ELMO的论文题目：`“Deep contextualized word representation”` 更能体现其精髓，`deep contextualized这个短语`。

在此之前的 `Word Embedding` 本质上是个**静态的方式**，所谓静态指的是训练好之后每个单词的表达就固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的 `Word Embedding` 不会跟着上下文场景的变化而改变，所以对于比如 `Bank` 这个词，它事先学好的 `Word Embedding` 中混合了几种语义 ，在应用中来了个新句子，即使从上下文中（比如句子包含 `money` 等词）明显可以看出它代表的是 `“银行”` 的含义，但是对应的Word Embedding内容也不会变，它还是**混合了多种语义**。

**ELMO的本质思想是**：我事先用语言模型学好一个单词的`Word Embedding`，此时多义词无法区分，不过这没关系。在实际使用`Word Embedding`的时候，单词已经具备了特定的上下文了，这个时候可以根据上下文单词的语义去**调整**单词的`Word Embedding`表示，这样经过调整后的`Word Embedding`更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。**所以ELMO本身是个根据当前上下文对`Word Embedding`动态调整的思路。**

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/bert/elmo.png" width="550" >

`ELMO`采用了**典型的两阶段**过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的网络各层的`Word Embedding`作为新特征补充到下游任务中。

上图展示的是其预训练过程，它的网络结构采用了双层双向LSTM，目前语言模型训练的任务目标是根据单词 $W_i$ 的上下文去正确预测单词 $W_i$ 之前的单词序列`Context-Before`称为上文，之后的单词序列`Context-After`称为下文。

* 图中左端的前向双层LSTM代表**正方向编码器**，输入的是从左到右顺序的除了预测单词外 $W_i$ 的上文 `Context-Before`；右端的逆向双层LSTM代表**反方向编码器**，输入的是从右到左的逆序的句子下文 `Context-After`；每个编码器的深度都是**两层LSTM叠加**。

这个网络结构其实在NLP中是很常用的。

使用这个网络结构利用大量语料做语言模型任务就能预先训练好这个网络，如果训练好这个网络后，输入一个新句子 $S_{new}$ ，句子中每个单词都能得到对应的三个`Embedding`:

- 最底层是单词的`Word Embedding`；
- 往上走是第一层双向`LSTM`中对应单词位置的`Embedding`，这层编码单词的**句法信息**更多一些；
- 再往上走是第二层`LSTM`中对应单词位置的`Embedding`，这层编码单词的**语义信息**更多一些。

也就是说，ELMO的预训练过程不仅仅学会单词的`Word Embedding`，还学会了**一个双层双向的LSTM网络结构**，而这两者后面都有用。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/bert/elmo-train.png" width="550" >

上面介绍的是ELMO的第一阶段：预训练阶段。那么预训练好网络结构后，**如何给下游任务使用呢**？比如我们的下游任务仍然是`QA`问题:

- 此时对于问句 $X$，我们可以先将句子 $X$ 作为预训练好的ELMO网络的输入，这样句子 $X$ 中每个单词在ELMO网络中都能获得对应的三个`Embedding`；
- 之后给予这三个`Embedding`中的每一个`Embedding`一个权重 $a$，这个权重可以学习得来，根据各自权重累加求和，将三个`Embedding`整合成一个；
- 然后将整合后的这个`Embedding`作为 $X$ 句在自己任务的那个网络结构中对应单词的输入，以此作为补充的新特征给下游任务使用。对于下游任务QA中的回答句子 $Y$ 来说也是如此处理。

**因为ELMO给下游提供的是每个单词的特征形式**，所以这一类预训练的方法被**称为 `“Feature-based Pre-Training”`**。



**:question: ELMO引入上下文动态调整单词的`Embedding`后多义词问题解决了吗？**

> 解决了，而且比我们期待的解决得还要好。
>
> 对于GloVe训练出的`Word Embedding`来说，多义词比如play，根据它的Embedding找出的最接近的其它单词大多数集中在体育领域，这很明显是因为训练数据中包含play的句子中体育领域的数量明显占优导致；而使用ELMO，根据上下文动态调整后的Embedding不仅能够找出对应的“演出”的相同语义的句子，而且还可以保证找出的句子中的play对应的词性也是相同的，这是超出期待之处。
>
> 之所以会这样，是因为我们上面提到过，第一层LSTM编码了很多句法信息，这在这里起到了重要作用。

:question:**ELMO有什么值得改进的缺点呢**？

> - 首先，一个非常明显的缺点在特征抽取器选择方面，ELMO**使用了LSTM而不是Transformer**，Transformer是谷歌在17年做机器翻译任务的“Attention is all you need”的论文中提出的，引起了相当大的反响，很多研究已经证明了**Transformer提取特征的能力是要远强于LSTM的**。
> - 另外一点，ELMO采取**双向拼接**这种融合特征的能力可能比Bert一体化的融合特征方式弱，但是，这只是一种从道理推断产生的怀疑，目前并没有具体实验说明这一点。



#### :four: GPT

GPT是`“Generative Pre-Training”`的简称，从名字看其含义是指的生成式的预训练。GPT也采用两阶段过程，第一个阶段是利用语言模型进行预训练，第二阶段通过`Fine-tuning`的模式解决下游任务。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/bert/gpt.png" width="350" >

上图展示了GPT的预训练过程，其实和ELMO是类似的，主要不同在于两点：

- 首先，特征抽取器不是用的RNN，而是用的**Transformer**，上面提到过它的特征抽取能力要强于RNN，这个选择很明显是很明智的；
- 其次，GPT的预训练虽然仍然是以语言模型作为目标任务，但是采用的是**单向的语言模型**，所谓“单向”的含义是指：语言模型训练的任务目标是根据 $W_i$ 单词的上下文去正确预测单词 $W_i$ ，$W_i$之前的单词序列 `Context-Before` 称为上文，之后的单词序列 `Context-After` 称为下文。

ELMO在做语言模型预训练的时候，预测单词 $W_i$ 同时使用了上文和下文，而GPT则只采用 `Context-Before` 这个单词的上文来进行预测，而抛开了下文。这个选择现在看不是个太好的选择，原因很简单，它**没有把单词的下文融合进来**，这限制了其在更多应用场景的效果，比如阅读理解这种任务，在做任务的时候是可以允许同时看到上文和下文一起做决策的。

如果预训练时候不把单词的下文嵌入到`Word Embedding`中，是很吃亏的，白白丢掉了很多信息。

---



### §9.3 BERT模型

Bert采用和GPT完全相同的两阶段模型，首先是语言模型**预训练**；其次是使用**`Fine-Tuning`模型解决下游任务**。和GPT的最主要不同在于在预训练阶段采用了类似ELMO的双向语言模型，即双向的Transformer，当然另外一点是语言模型的数据规模要比GPT大。模型结构如下：

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/bert/bert-gpt-elmo.png" width="750" >

对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向rnn和双向rnn的区别，直觉上来讲效果会好一些。

对比ELMo，虽然都是“双向”，但目标函数其实是不同的。

- ELMo是分别以 $P(w_i|w_1,...,w_{i-1})$ 和 $P(w_i|w_{i+1},...,w_n)$ 作为目标函数，独立训练处两个representation然后拼接
- 而BERT则是以作为目标函数 $P(w_i|w_1,...,w_{i-1},w_{i+1},..,w_n)$ 训练LM。

BERT预训练模型分为以下三个步骤：**Embedding、Masked LM、Next Sentence Prediction**

#### Embedding



#### Masked LM



#### Next Sentence Prediction

