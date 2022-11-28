## TextRNN



### §5.1 TextRNN简介

文本分类任务中，**TextCNN可以用来提取句子中类似N-Gram的关键信息，适合短句子文本**。尽管TextCNN能够在很多任务里面能有不错的表现，但TextCNN有个最大问题是**固定filter_size的视野**:

- 一方面无法建模更长的序列信息
- 另一方面 filter_size 的超参调节也很繁琐

**CNN本质是做文本的特征表达工作**，而自然语言处理中更常用的是递归神经网络(RNN, Recurrent Neural Network），**能够更好的表达上下文信息,TextRNN擅长捕获更长的序列信息**。

- [x] 具体到文本分类任务中，从某种意义上可以理解为可以**捕获变长、单向的N-Gram信息**（Bi-LSTM可以是双向)。
- [x] 普通RNN在处理较长文本时会出现**梯度消失**问题，因此文本中RNN选用LSTM进行实验。

RNN是自然语言处理领域常见的一个标配网络，在序列标注/命名体识别/seq2seq模型等很多场景都有应用

[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf) 文中介绍了RNN用于分类问题的设计，下图是RNN用于网络结构原理示意图，示例中是利用最后一个词的结果，可以看做是包含了前面所有词语的信息，然后直接接全连接层softmax输出了。

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextRNN/RNN.png" width="550" />

---



### §5.2 TextRNN具体结构：LSTM & Bi-LSTM

#### LSTM网络

##### :one:输入输出

LSTM网络（图右）与普通RNN的主要输入输出区别

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextRNN/lstm.png" width="450" />

* RNN有一个传递状态$h^t$
* LSTM有两个传递状态，一个cell state $c^t$ 和一个hidden state $h^t$ （RNN的$h^t$对应于LSTM的$c^t$）
  * 其中对于传递下去的从 $c^t$ 改变得很慢，通常输出的 $c^t$ 是上一个状态传过来的 $c^{t-1}$ 加上一些数值
  *  而 $h^t$ 则在不同的节点下有很大的区别

##### :two:状态计算

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextRNN/lstm2.png" width="425" />

使用LSTM的当前输入 $x^t$ 和上一个状态传递下来的 $h^{t-1}$ 拼接训练得到四个状态

* $z^f$, $z^i$, $z^o$ 是由拼接向量乘以相应的权重矩阵之后，再通过 $sigmoid$ 激活函数转换成0到1之间的数值，来作为一种门控状态
* $z$ 则是将结果通过一个 $tanh$ 激活函数将拼接向量和权重矩阵相乘的结果转换成-1到1的值，使用 $tanh$ 是因为将结果作为输入数据而非门控信息

**:three:内部结构**

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextRNN/lstm3.png" width="500" />

其中 $\odot$ 代表Hadamard Product，操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的， $\oplus$代表矩阵加法

LSTM内部主要有三个阶段

* **遗忘阶段**
  * 对上一个节点传进来的输入进行选择性遗忘，“忘记不重要的，记住重要的”
  * 使用 $z^f$ 作为**遗忘门控**(f for forget)，来控制上一个状态 $c^{t-1}$ 那些需要遗忘，那些需要留下
* **选择记忆阶段**
  * 该阶段对输入$x^t$ 进行有选择地记忆，哪些重要则着重记录下来，不重要的少记录一些，当前输入的内容由 $z$ 表示，而选择的门控信号则由 $z^i$ 来进行控制 (i for information)
  * 遗忘阶段和选择记忆阶段得到的结果相加即可得到传输给下一个状态的 $c^t$
  * $c^t = z^f \odot c^{t-1} \oplus z^i \odot z$
* **输出阶段**
  * 该阶段决定那些会被当成当前状态的输出，通过 $z^o $来控制，还对上一个阶段得到的 $c^t$ 进行了放缩（使用一个 $tanh$ 激活函数进行变化）
  * 输出 $y^t$ 也是通过 $h^t$ 变换得到，类似于普通RNN



#### Bi-LSTM

Bi-LSTM是LSTM的改进版本，将单向RNN结构改成了双向RNN，希望不仅能考虑正向编码的信息，也能考虑反向编码的信息,模型结构如下图所示

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/TextRNN/bi-lstm.png" width="425" />

- [x] 与LSTM不同的是，在RNN部分使用了Bi-LSTM进行信息提取，Bi-LSTM层中，内部有两个LSTM，分别为 $Forward$ 层和 $Backward$ 层，表示前向与后向
- [x] 每个LSTM设定输出的维度为 $embedingSize$ 维向量，在 $Forward$ 层从 $1$ 时刻到 $t$ 时刻正向计算一遍，得到并保存每个时刻向前隐含层的输出。
- [x] 在 $Backward$ 层沿着时刻 $t$ 到时刻 $1$ 反向计算一遍，得到并保存每个时刻向后隐含层的输出。
- [x] 最后在每个时刻结合 $Forward$ 层和 $Backward$ 层的相应时刻输出的结果进行拼接得到最终的输出。因此输出的维度为 $embedingSize*2$ 维。

---



### §5.3 TextRNN总结

TextRNN的结构非常灵活，可以任意改变。比如把LSTM单元替换为GRU单元，把双向改为单向，添加dropout或BatchNormalization以及再多堆叠一层等等。

TextRNN在文本分类任务上的效果非常好，与TextCNN不相上下，但RNN的**训练速度相对偏慢**，一般2层就已经足够多了。