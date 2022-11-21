## TextCNN



#### §4.1 TextCNN简介

我们可以将文本当作**一维图像**，从而可以用**⼀维卷积神经网络**来捕捉临近词之间的关联。

##### 核心思想

TextCNN是Yoon Kim在2014年于论文 [Convolutional Naural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) 中提出的文本分类模型，开创了**用CNN编码n-gram特征**的先河。

- 我们知道fastText 中的网络结构是完全没有考虑词序信息的，而它用的 n-gram 特征 trick 恰恰说明了**局部序列信息**的重要意义。
- 卷积神经网络(CNN Convolutional Neural Network)最初在图像领域取得了巨大成功，**CNN原理的核心点在于可以捕捉局部相关性(局部特征)，对于文本来说，局部特征就是由若干单词组成的滑动窗口，类似于N-gram。**
- **卷积神经网络的优势在于能够自动地对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息。**

与传统图像的CNN网络相比, TextCNN 在网络结构上没有任何变化（甚至更加简单了）。从下图可以看出TextCNN 其实只有一层卷积, 一层max-pooling, 最后将输出外接softmax来n分类。

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextCNN/TextCNN.png" width = "450" />