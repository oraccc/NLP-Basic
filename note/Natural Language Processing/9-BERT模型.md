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
