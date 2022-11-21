## TextCNN



### §4.1 TextCNN简介

我们可以将文本当作**一维图像**，从而可以用**⼀维卷积神经网络**来捕捉临近词之间的关联。

#### 核心思想

TextCNN是Yoon Kim在2014年于论文 [Convolutional Naural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) 中提出的文本分类模型，开创了**用CNN编码n-gram特征**的先河。

- [x] 我们知道fastText 中的网络结构是完全没有考虑词序信息的，而它用的 n-gram 特征 trick 恰恰说明了**局部序列信息**的重要意义。
- [x] 卷积神经网络(CNN Convolutional Neural Network)最初在图像领域取得了巨大成功，**CNN原理的核心点在于可以捕捉局部相关性(局部特征)，对于文本来说，局部特征就是由若干单词组成的滑动窗口，类似于N-gram。**
- [x] **卷积神经网络的优势在于能够自动地对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息。**

与传统图像的CNN网络相比, TextCNN 在网络结构上没有任何变化（甚至更加简单了）。从下图可以看出TextCNN 其实只有一层卷积, 一层max-pooling, 最后将输出外接softmax来n分类。

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextCNN/TextCNN2.png" width = "700" />

#### 与传统CNN网络的不同

与图像当中CNN的网络相比，textCNN 最大的不同便是在输入数据的不同：

- 图像是二维数据, 图像的卷积核都是**二维**的，是从左到右, 从上到下进行滑动来进行特征抽取。

- 自然语言是一维数据,**textCNN使用一维卷积，即 $filter\_size*embedding\_dim$，有一个维度和embedding相等**。

   > 虽然经过word-embedding 生成了二维词向量，但是对词向量做从左到右滑动来进行卷积没有意义. 比如 "今天" 对应的向量[0, 0, 0, 0, 1], 按窗口大小为 1* 2 从左到右滑动得到[0,0], [0,0], [0,0], [0, 1]这四个向量, 对应的都是"今天"这个词汇, 这种滑动没有帮助.

---



### §4.2 TextCNN模型具体结构

TextCNN的详细过程与原理图如下所示

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextCNN/TextCNN.png" width = "500" />

#### 详细过程

第一层是图中最左边的7乘5的句子矩阵，每行是词向量，$dim=5$，这个可以类比为图像中的原始像素点了。以1个样本为例，整体的前向逻辑是：

- [x] 对词进行embedding，得到 $[seq\_length, embedding\_dim]$
- [x] 用N个卷积核，得到N个 $seq\_length - filter\_size + 1$ 长度的一维feature map
- [x] 对feature map进行max-pooling（**因为是时间维度的，也称max-over-time pooling**），得到N个 $1 * 1$ 的数值，这样不同长度句子经过pooling层之后都能变成定长的表示了，然后拼接成一个N维向量，作为文本的句子表示
- [x] 最后接一层全连接的 softmax 层，将N维向量压缩到类目个数的维度，输出每个类别的概率。

#### 实现细节

##### :one:特征

这里的特征就是词向量，有**静态（static）**和**非静态（non-static）**方式。

* static方式采用比如word2vec预训练的词向量，训练过程不更新词向量，实质上属于迁移学习了，特别是数据量比较小的情况下，采用静态的词向量往往效果不错。
* non-static则是在训练过程中**更新词向量**。
  * 推荐的方式是 non-static 中的 fine-tunning方式，它是以预训练（pre-train）的word2vec向量初始化词向量，训练过程中调整词向量，能加速收敛
  * 当然，如果有充足的训练数据和资源，直接随机初始化词向量效果也是可以的

##### :two:通道(Channels)

图像中可以利用 (R, G, B) 作为不同channel，而**文本的输入的channel通常是不同方式的embedding方式**（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。

根据原论文作者的描述, 一开始引入channel 是希望防止过拟合(通过保证学习到的vectors 不要偏离输入太多)来在小数据集合获得比单channel更好的表现，后来发现其实**直接使用正则化效果**更好。

对于channel在TextCNN 是否有用, 从论文的实验结果来看多channels并没有明显提升模型的分类能力, 七个数据集上的五个数据集单channel的TextCNN 表现都要优于多channels的TextCNN



我们在这里也介绍一下论文中四个model 的不同：

> * CNN-rand (单channel)
>   * 设计好 embedding_size 这个 Hyperparameter 后, 对不同单词的向量作**随机初始化**, 后续BP的时候作调整
> * CNN-static(单channel)
>   *  拿 **pre-trained vectors** from word2vec, FastText or GloVe 直接用, 训练过程中不再调整词向量
> * CNN-non-static(单channel)
>   * **pre-trained vectors + fine tuning** , 即拿word2vec训练好的词向量初始化, 训练过程中再对它们微调
> * CNN-multiple channel(多channels)
>   * 类比于图像中的RGB通道, 这里也可以用 static 与 non-static 搭两个通道来做

##### :three:一维卷积(conv-1d)

##### :four:池化(Max-pooling) 

##### :five:全连接+softmax层