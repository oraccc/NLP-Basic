## TextCNN

- [TextCNN](#textcnn)
  - [§4.1 TextCNN简介](#41-textcnn简介)
    - [核心思想](#核心思想)
    - [与传统CNN网络的不同](#与传统cnn网络的不同)
  - [§4.2 TextCNN模型具体结构](#42-textcnn模型具体结构)
    - [详细过程](#详细过程)
    - [实现细节](#实现细节)
      - [:one:特征](#one特征)
      - [:two:通道(Channels)](#two通道channels)
      - [:three:一维卷积(conv-1d)](#three一维卷积conv-1d)
      - [:four:池化(Max-pooling)](#four池化max-pooling)
      - [:five:全连接+softmax层](#five全连接softmax层)
  - [§4.3 TextCNN模型的优缺点](#43-textcnn模型的优缺点)



### §4.1 TextCNN简介

> :memo: [Jupyter Notebook 代码](https://github.com/oraccc/NLP-Basic/blob/master/code/Natural%20Language%20Processing/4-textcnn.ipynb)

我们可以将文本当作**一维图像**，从而可以用**⼀维卷积神经网络**来捕捉临近词之间的关联。

#### 核心思想

TextCNN是Yoon Kim在2014年于论文 [Convolutional Naural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) 中提出的文本分类模型，开创了**用CNN编码n-gram特征**的先河。

- [x] 我们知道fastText 中的网络结构是完全没有考虑词序信息的，而它用的 n-gram 特征 trick 恰恰说明了**局部序列信息**的重要意义。
- [x] 卷积神经网络(CNN Convolutional Neural Network)最初在图像领域取得了巨大成功，**CNN原理的核心点在于可以捕捉局部相关性(局部特征)，对于文本来说，局部特征就是由若干单词组成的滑动窗口，类似于N-gram。**
- [x] **卷积神经网络的优势在于能够自动地对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息。**

与传统图像的CNN网络相比, TextCNN 在网络结构上没有任何变化（甚至更加简单了）。从下图可以看出TextCNN 其实只有一层卷积, 一层max-pooling, 最后将输出外接softmax来n分类。

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/textcnn/TextCNN2.png" width = "700" />

#### 与传统CNN网络的不同

与图像当中CNN的网络相比，textCNN 最大的不同便是在输入数据的不同：

- 图像是二维数据, 图像的卷积核都是**二维**的，是从左到右, 从上到下进行滑动来进行特征抽取。

- 自然语言是一维数据,**textCNN使用一维卷积，即 $filterSize*embeddingDim$，有一个维度和embedding相等**。

   > 虽然经过word-embedding 生成了二维词向量，但是对词向量做从左到右滑动来进行卷积没有意义. 比如 "今天" 对应的向量[0, 0, 0, 0, 1], 按窗口大小为 1* 2 从左到右滑动得到[0,0], [0,0], [0,0], [0, 1]这四个向量, 对应的都是"今天"这个词汇, 这种滑动没有帮助.

---



### §4.2 TextCNN模型具体结构

TextCNN的详细过程与原理图如下所示

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/textcnn/TextCNN.png" width = "500" />

#### 详细过程

第一层是图中最左边的7乘5的句子矩阵，每行是词向量，$dim=5$，这个可以类比为图像中的原始像素点了。以1个样本为例，整体的前向逻辑是：

- [x] 对词进行embedding，得到 $[seqLength, embeddingDim]$
- [x] 用N个卷积核，得到N个 $seqLength - filterSize + 1$ 长度的一维feature map
- [x] 对feature map进行max-pooling（**因为是时间维度的，也称max-over-time pooling**），得到N个 $1 * 1$ 的数值，这样不同长度句子经过pooling层之后都能变成定长的表示了，然后拼接成一个N维向量，作为文本的句子表示
- [x] 最后接一层全连接的 softmax 层，将N维向量压缩到类目个数的维度，输出每个类别的概率。



#### 实现细节

##### :one:特征

这里的特征就是词向量，有**静态**和**非静态**方式。

* static方式采用比如word2vec预训练的词向量，训练过程不更新词向量，实质上属于迁移学习了，特别是数据量比较小的情况下，采用静态的词向量往往效果不错。
* :star2:non-static则是在训练过程中**更新词向量**。
  * 推荐的方式是 non-static 中的 fine-tunning方式，它是以预训练（pre-train）的word2vec向量初始化词向量，训练过程中调整词向量，能加速收敛
  * 当然，如果有充足的训练数据和资源，直接随机初始化词向量效果也是可以的

##### :two:通道(Channels)

图像中可以利用 (R, G, B) 作为不同channel，而**文本的输入的channel通常是不同方式的embedding方式**（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。

根据原论文作者的描述, 一开始引入channel 是希望防止过拟合(通过保证学习到的vectors 不要偏离输入太多)来在小数据集合获得比单channel更好的表现，后来发现其实**直接使用正则化效果**更好。

对于channel在TextCNN 是否有用, 从论文的实验结果来看多channels并没有明显提升模型的分类能力, 七个数据集上的五个数据集单channel的TextCNN 表现都要优于多channels的TextCNN

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/textcnn/experiment.png" width = "700" />

论文中四个model 的不同：

> * CNN-rand (单channel)
>   * 设计好 embedding_size 这个 Hyperparameter 后, 对不同单词的向量作**随机初始化**, 后续BP的时候作调整
> * CNN-static(单channel)
>   *  拿 **pre-trained vectors** from word2vec, FastText or GloVe 直接用, 训练过程中不再调整词向量
> * CNN-non-static(单channel)
>   * **pre-trained vectors + fine tuning** , 即拿word2vec训练好的词向量初始化, 训练过程中再对它们微调
> * CNN-multiple channel(多channels)
>   * 类比于图像中的RGB通道, 这里也可以用 static 与 non-static 搭两个通道来做

##### :three:一维卷积(conv-1d)

图像是二维数据，经过词向量表达的文本为一维数据，因此在TextCNN卷积用的是一维卷积。一维卷积带来的问题是需要设计通过不同 filter_size 的 filter 获取不同宽度的视野。**一般来说在卷积之后会跟一个激活函数，例如ReLU或tanh**

:star2:激活函数的可以为线性模型**引入非线性因素**，从而**解决线性模型难以解决的问题**

##### :four:池化(Max-pooling) 

这里使用的是max-pooling，从feature map中选出最大的一个值。也可以改成 **(dynamic) k-max pooling ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息**。分类任务对细粒度语义的要求不高，只抓住最大特征就好了

> 例子：比如在情感分析场景，有以下的句子
>
> “ 我觉得这个地方景色还不错，但是人也实在太多了 ”
>
> * 虽然前半部分体现情感是正向的，全局文本表达的是偏负面的情感，利用 k-max pooling能够很好捕捉这类信息。

:star2:max-pooling 在保持主要特征的情况下, 大大降低了参数的数目，好处有如下几点:

- 这种Pooling方式可以解决可变长度的句子输入问题（因为不管Feature Map中有多少个值，只需要提取其中的最大值）
- 降低了过拟合的风险, feature map = [1, 1, 2] 或者[1, 0, 2] 最后的输出都是[2], 表明开始的输入即使有轻微变形, 也不影响最后的识别
- 参数减少, 进一步加速计算

关于**平移不变性**(图片有个字母A, 这个字母A 无论出现在图片的哪个位置, 在CNN的网络中都可以识别出来)，**卷积核的权值共享才能带来平移不变性**。max-pooling的原理主要是从多个值中取一个最大值，做不到这一点。

:star2:**CNN能够做到平移不变性，是因为在滑动卷积核的时候，使用的卷积核权值是保持固定的(权值共享)**, 假设这个卷积核被训练的就能识别字母A, 当这个卷积核在整张图片上滑动的时候，当然可以把整张图片的A都识别出来。

##### :five:全连接+softmax层

我们将 max-pooling的结果通过全连接的方式，连接一个softmax层，softmax层可根据任务的需要设置（通常反映着最终类别上的概率分布）。为了防止过拟合，在倒数第二层的全连接部分上使用dropout技术

:star2:对**dropout**的理解

* **dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作**，不工作的那些节点可以暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了，它是**防止模型过拟合**的一种常用的trick （防止模型学的过好）。

同时对全连接层上的权值参数给予**L2正则化**的限制。这样做的好处是防止隐藏层单元自适应（或者对称），从而减轻过拟合的程度。

:star2:对于**L2正则化**的理解：

* 从直观角度考虑
    * 过拟合可以理解为模型的复杂度过高。对于线性回归问题，也就是$w$参数较大。对于一个线性模型：  $y=wx+b$ 。如果 $w = 0$，则 $y$ 恒等于 $b$ ，此时模型几乎没有拟合能力（欠拟合）。
    * 通过增加一个关于w的惩罚项，将学习机向 $w = 0$ 的方向进行引导（因为单纯从 $||w||^2$ 这一项看，它越小，$L$ 才能越小），以此来克服过拟合。
* 从**病态矩阵**的角度考虑
    * 对于原始损失函数，其解析解为： $w=(X^TX)^{−1}X^Ty$ 。事实上这里存在两个问题
    * 一个是 $X^TX$ 可能不可逆，此时说明 $X^TX$ 中有特征值为0，发生场景为 $X$ 的维度比样本数还大。
    * 另一个问题是即使 $X^TX$ 可逆，但是这个矩阵是病态的，也就是说如果 $y$ 存在很小的波动， 被 $(X^TX)^{−1}$ 乘了以后，结果 $w$ 都会发生很大的变化。那么我们计算的这个$w$就非常的不稳定，并不是一个好的模型，发生场景为$X$的维度比样本数差不多大小。
    * 判断一个矩阵是不是病态矩阵，可以通过计算矩阵的条件数。条件数等于矩阵的最大奇异值和最小奇异值之比。如果矩阵 $X^TX$ 存在很小的奇异值，那么它的逆就存在很大的奇异值，这样对 $y$ 中的微小变化会放大很多。所以我们的目标就是去除 $X^TX$ 中极小的奇异值。
    * 加了L2正则项之后，我们的解析解变为：$w=(X^TX+λI)^{−1}X^Ty$ 。也就是给 $X^TX$ 中的所有奇异值加上一个 $λ$ ，可以确保奇异值不会太小，而导致再求逆后，奇异值变的极大。这样有效的解决了病态矩阵的问题。**过拟合的实质可以看作由于病态矩阵的存在，如果 $y$ 有一点波动，整个模型需要大幅度调整**。解决了病态矩阵问题，就解决了过拟合。

---



### §4.3 TextCNN模型的优缺点

- TextCNN**模型简单, 训练速度快，效果不错**。是很适合中短文本场景的强baseline
- TextCNN但不太适合长文本，因为卷积核尺寸通常不会设很大，**无法捕获长距离特征**
- 同时max-pooling也存在局限，会**丢掉一些有用特征**
- 另外再仔细想的话，TextCNN和传统的n-gram词袋模型本质是一样的，它的好效果很大部分来自于词向量的引入，解决了**词袋模型的稀疏性问题**
