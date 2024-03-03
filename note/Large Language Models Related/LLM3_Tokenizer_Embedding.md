## LLM的Tokenizer与Embedding

### Tokenizer

tokenizer总体上做三件事情：

1. **分词**
   tokenizer将字符串分为一些sub-word token string，再将token string映射到id，并保留来回映射的mapping。从string映射到id为tokenizer encode过程，从id映射回token为tokenizer decode过程。映射方法有多种，例如BERT用的是**WordPiece**，GPT-2和RoBERTa用的是**BPE**等等。
2. **扩展词汇表**
   部分tokenizer会用一种统一的方法将训练语料出现的且词汇表中本来没有的token加入词汇表。对于不支持的tokenizer，用户也可以手动添加。
3. **识别并处理特殊token**
   特殊token包括[MASK], <|im_start|>, \<sos>, \<s>等等。tokenizer会将它们加入词汇表中，并且保证它们在模型中**不被切成sub-word**，而是完整保留。

#### 分词粒度

我们首先来看一下几种不同的分词粒度。

最直观的分词是**单词分词法（word base）**。单词分词法将一个word作为最小元，也就是根据空格或者标点分词。举例来说，Today is Sunday用word-base来进行分词会变成['Today', 'is', 'Sunday']。

最详尽的分词是**单字分词法（character-base）**。单字分词法会穷举所有出现的字符，所以是最完整的。在上面的例子中，单字分词法会生成['T', 'o', 'd', ..., 'a', 'y']。

另外还有一种最常用的、介于两种方法之间的分词法叫**子词分词法**，会把上面的句子分成最小可分的子词['To', 'day', 'is', 'S', 'un', 'day']。子词分词法有很多不同取得最小可分子词的方法，例如**BPE（Byte-Pair Encoding，字节对编码法），WordPiece，SentencePiece，Unigram**等等。

#### GPT族：Byte-Pair Encoding (BPE)

```
1. 统计输入中所有出现的单词并在每个单词后加一个单词结束符</w> -> ['hello</w>': 6, 'world</w>': 8, 'peace</w>': 2]
2. 将所有单词拆成单字 -> {'h': 6, 'e': 10, 'l': 20, 'o': 14, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
3. 合并最频繁出现的单字(l, o) -> {'h': 6, 'e': 10, 'lo': 14, 'l': 6, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
4. 合并最频繁出现的单字(lo, e) -> {'h': 6, 'lo': 4, 'loe': 10, 'l': 6, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
5. 反复迭代直到满足停止条件
```

显然，这是一种贪婪的算法。在上面的例子中，'loe'这样的子词貌似不会经常出现，但是当语料库很大的时候，诸如est，ist，sion，tion这样的特征会很清晰地显示出来。

在获得子词词表后，就可以将句子分割成子词了，算法见下面的例子：

```
# 给定单词序列
["the</w>", "highest</w>", "mountain</w>"]

# 从一个很大的corpus中排好序的subword表如下
# 长度 6         5           4        4         4       4          2
["errrr</w>", "tain</w>", "moun", "est</w>", "high", "the</w>", "a</w>"]

# 迭代结果
"the</w>" -> ["the</w>"]
"highest</w>" -> ["high", "est</w>"]
"mountain</w>" -> ["moun", "tain</w>"]
```

注意，在上述算法执行后，如果句子中仍然有子字符串没被替换, 但所有subword都已迭代完毕，则将剩余的子词替换为特殊token，如\<unk> 。从这里大家也可以发现了，**原则上<unk>这个token出现的越少越好，所以我们也往往用\<unk>的数量来评价一个tokenizer的好坏程度，这个token出现的越少，tokenizer的效果往往越好。**

管中窥豹，根据BPE算法，我们可以发现，tokenizer基本上是无法并行的，因为存在大量if-else的branch。学过GPU Programming的同学应该知道，conditional branch越多，GPU提供的加速越有限，有时候还会造成负加速，因为数据传输有很大开销。这就是为什么在tokenizing的时候，我们看到GPU util都是0。

#### BERT族：Word-Piece

Word-Piece和BPE非常相似，BPE使用**出现最频繁的组合**构造子词词表，而WordPiece使用**出现概率最大的组合**构造子词词表。换句话说，WordPiece每次选择合并的两个子词，通常在语料中以相邻方式同时出现。比如说 P(ed) 的概率比P(e) + P(d)单独出现的概率更大（可能比他们具有最大的互信息值），也就是两个子词在语言模型上具有较强的关联性。这个时候，Word-Piece会将它们组合成一个子词。

BERT在句首加上了[CLS]，句尾加上了[SEP]

#### 多语言支持：Sentence-Piece

大家在使用HF的时候有时候会提示安装Sentence-Piece，这个包其实是HF里面大量模型会调用的包，例如ALBERT，XLM-RoBERTa和T5：

这个包主要是为了多语言模型设计的，它做了两个重要的转化：

1. 以unicode方式编码字符，将所有的输入（英文、中文等不同语言）都转化为unicode字符，解决了多语言编码方式不同的问题。
2. 将空格编码为‘\_’， 如'New York' 会转化为['\_', 'New', '_York']，这也是为了能够处理多语言问题，比如英文解码时有空格，而中文没有， 类似这种语言区别。

##### 词汇表不全问题

但是，也是因为这两个转化，SentencePiece的tokenizer往往会出现词汇表不全的问题。下面是部分SentencePiece中可能出现的问题

如果某个token被识别成\<unk>，那它就无法与其他也被识别成\<unk>的token区分开来。例如在训练的时候有大量{hello world}的样本，在输出的时候就会变成\<unk> hello world \<unk> 的样本。

这些问题不存在于WordPiece中。这是因为SentencePiece需要对多语言情况进行优化，有些token迫不得已要被删掉。想要加上某些本来tokenizer中不存在的token，可以使用add_tokens()方法。

#### 各路语言模型中的tokenizer

各大LM用的tokenizer和对应的词汇表大小如下

<img src="..\..\img\llm-basic\llm_tokenizer.png" alt="图片" style="zoom: 67%;" />

### Embedding

tokenize完的下一步就是将token的one-hot编码转换成更dense的embedding编码。在ELMo之前的模型中，embedding模型很多是单独训练的，而ELMo之后则爆发了直接将embedding层和上面的语言模型层共同训练的浪潮（ELMo的全名就是Embeddings from Language Model）。不管是哪种方法，Embedding层的形状都是一样的。

在HuggingFace中，seq2seq模型往往是这样调用的：

```python
input_ids = tokenizer.encode('Hello World!', return_tensors='pt')
output = model.generate(input_ids, max_length=50)
tokenizer.decode(output[0])
```

上面的代码主要涉及三个操作：tokenizer将输入encode成数字输入给模型，模型generate出输出数字输入给tokenizer，tokenizer将输出数字decode成token并返回。

例如，如果我们使用T5TokenizerFast来tokenize 'Hello World!'，则：

1. tokenizer会将token序列 ['Hello', 'World', '!'] 编码成数字序列[8774, 1150, 55, 1]，也就是['Hello', 'World', '!', '']，在句尾加一个表示句子结束。
2. 这四个数字会变成四个one-hot向量，例如8774会变成[0, 0, ..., 1, 0, 0..., 0, 0]，其中向量的index为8774的位置为1，其他位置全部为0。假设词表里面一共有30k个可能出现的token，则向量长度也是30k，这样才能保证出现的每个单词都能被one-hot向量表示。
3. 也就是说，一个形状为 (4) 的输入序列向量，会变成形状为 (4, 30k) 的输入one-hot向量。为了将每个单词转换为一个word embedding，每个向量都需要被被送到embedding层进行dense降维。
4. 现在思考一下，多大的矩阵可以满足这个要求？没错，假设embedding size为768，则矩阵的形状应该为(30k, 768) ，与BERT的实现一致：

```
BertForSequenceClassification(
  (bert):BertModel(
    (embeddings):BertEmbeddings(
      (word_embeddings):Embedding(30522，768，padding_idx=0)
      (position_embeddings):Embedding(512，768) 
      (token_type_embeddings):Embedding(2，768)
      (LayerNorm): LayerNorm((768,)，eps=1e-12，elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False) 
    )
  )
)
```

#### 理解Embedding矩阵

**Embedding矩阵的本质就是一个查找表**。由于**输入向量是one-hot的，embedding矩阵中有且仅有一行被激活。**行间互不干扰。

> 假设词汇表一共有6个词，则one-hot表示的长度为6。现在我们有三个单词组成一个句子，则输入矩阵的形状为 (3, 6)。然后我们学出来一个embedding矩阵，根据上面的推导，如果我们的embedding size为4，则embedding矩阵的形状应该为(6, 4)。这样乘出来的输出矩阵的形状应为(3, 4)。

embedding矩阵的本质是一个查找表，每个单词会定位这个表中的某一行，而这一行就是这个单词学习到的在嵌入空间的语义