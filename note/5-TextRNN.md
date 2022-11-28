## TextRNN



### §5.1 TextRNN简介

文本分类任务中，**TextCNN可以用来提取句子中类似N-Gram的关键信息，适合短句子文本**。尽管TextCNN能够在很多任务里面能有不错的表现，但TextCNN有个最大问题是**固定filter_size的视野**:

- 一方面无法建模更长的序列信息
- 另一方面 filter_size 的超参调节也很繁琐

**CNN本质是做文本的特征表达工作**，而自然语言处理中更常用的是递归神经网络(RNN, Recurrent Neural Network），**能够更好的表达上下文信息,TextRNN擅长捕获更长的序列信息**。

- [x] 具体到文本分类任务中，从某种意义上可以理解为可以**捕获变长、单向的N-Gram信息**（Bi-LSTM可以是双向)。
- [x] 普通RNN在处理较长文本时会出现**梯度消失**问题，因此文本中RNN选用LSTM进行实验。

RNN是自然语言处理领域常见的一个标配网络，在序列标注/命名体识别/seq2seq模型等很多场景都有应用，[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf) 文中介绍了RNN用于分类问题的设计，下图是RNN用于网络结构原理示意图，示例中是利用最后一个词的结果，可以看做是包含了前面所有词语的信息，然后直接接全连接层softmax输出了。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextRNN/RNN.png" width="550" />



### §5.2 LSTM & Bi-LSTM

#### LSTM网络

