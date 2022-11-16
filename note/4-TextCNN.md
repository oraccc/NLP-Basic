## TextCNN



#### §4.1 TextCNN简介

我们可以将文本当作**一维图像**，从而可以用**⼀维卷积神经网络**来捕捉临近词之间的关联。

我们之前提前CNN时，通常会认为是属于CV领域，用于计算机视觉方向的工作，但是在2014年，Yoon Kim针对CNN的输入层做了一些变形，提出了文本分类模型TextCNN。与传统图像的CNN网络相比, TextCNN 在网络结构上没有任何变化(甚至更加简单了)。从下图可以看出TextCNN 其实只有一层卷积, 一层max-pooling, 最后将输出外接softmax来n分类。

<img src = "https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextCNN/TextCNN.png" width = "450" />