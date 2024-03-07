LLM 训练所需的数据来源大体上可以分为**通用数据**和**专业数据**两大类。

通用数据（General Data）包括网页、图书、新闻、对话文本等内容。通用数据具有规模大、多样性和易获取等特点，因此可以支持 LLM 的构建语言建模和泛化能力。

专业数据（Specialized Data）包括多语言数据、科学数据、代码以及领域特有资料等数据。通过在预训练阶段引入专业数据可以有效提供 LLM 的任务解决能力。

### 通用数据

通用数据在 LM 训练数据中占比通常非常高，主要包括网页、书籍、对话文本等类型，为大模型提供了大规模且多样的训练数据。

**网页（Webpage）是通用数据中数量最大的一类。**随着互联网的大规模普及，人们通过网站、论坛、博客、APP 等各种类型网站和应用，创造了海量的数据。因此，**如何过滤和处理网页以提升高质量数据的比例，对 LLM 训练来说非常重要。**

相较于其他语料库，**书籍也是最重要的，甚至是唯一的长文本书面语的数据来源。**书籍提供了完整的句子和段落，使得语言模型可以学习到上下文之间的联系。这对于模型理解句子中的复杂结构、逻辑关系和语义连贯性非常重要。书籍涵盖了各种文体和风格，包括小说、科学著作、历史记录等等。通过使用书籍数据训练 LM，可以使模型学习到不同的写作风格和表达方式，提高 LLM 在各种文本类型上的能力。

对话数据（Conversation Text）是指包含两个或更多参与者之间交流的文本内容。对话数据包含书面形式的对话、聊天记录、论坛帖子、社交媒体评论等。当前的一些研究也表明，对话数据可以有效增强 LM 的对话能力，并潜在地提高其在多种问答任务上的表现

### 专业数据

多语言数据（Multilingual Text）对于增强 LLM 语言理解和生成多语言能力具有至关重要的作用。

科学文本（Scientific Text）包括教材、论文、百科以及其他相关资源。这些数据对于提升 LLM 在理解科学知识方面具有重要作用

代码（Code）数据是进行程序生成任务所必须的训练数据。最近的研究和 ChatGPT 的结果表明，通过在大量代码上进行预训练，LLM 可以有效提升代码生成的效果。

### 数据处理

大语言模型的相关研究表明，数据质量对于模型的影响非常大。因此在收集到各类型数据之后，需要对数据进行处理，去除低质量数据、重复数据、有害信息、个人隐私等内容。典型的数据处理过程如下图所示，主要包含**质量过滤、冗余去除、隐私消除、词元切分**这四个步骤

#### 词元切分

传统的自然语言处理通常以单词为基本处理单元，模型都依赖预先确定的词表 **V**，在编码输入词序列时，这些词表示模型只能处理词表中存在的词。因此，在使用中，如果遇到不在词表中的未登录词，模型无法为其生成对应的表示，只能给予这些**未登录词（Out-of-vocabulary，OOV）**一个默认的通用表示。

在深度学习模型中，词表示模型会预先在词表中加入一个默认的“[UNK]”（unknown）标识，表示未知词，并在训练的过程中将 [UNK] 的向量作为词表示矩阵的一部分一起训练，通过 引入某些相应机制来更新 [UNK] 向量的参数。

在使用时，对于全部的未登录词，都使用 [UNK] 的向量作为这些词的表示向量。

此外，**基于固定词表的词表示模型对词表大小的选择比较敏感。当词表大小过小时，未登录词的比例较高，影响模型性能。而当词表大小过大时，大量低频词出现在词表中，而这些词的词向量很难得到充分学习。**

理想模式下，词表示模型应能覆盖绝大部分的输入词，并避免词表过大所造成的数据稀疏问题。

**为了缓解未登录词问题，一些工作通过利用亚词级别的信息构造词表示向量。**一种直接的解决思路是为输入建立字符级别表示，并通过字符向量的组合来获得每个单词的表示，以解决数据稀疏问题。然而，**单词中的词根、词缀等构词模式往往跨越多个字符，基于字符表示的方法很难学习跨度较大的模式。**为了充分学习这些构词模式，研究人员们提出了**子词词元化（Subword Tokenization）**方法，试图缓解上文介绍的未登录词问题。

**词元表示模型**会维护一个词元词表，其中既存在**完整的单词**，也存在形如“c”, “re”, “ing”等单词部分信息，称为**子词**。

**词元表示模型对词表中的每个词元计算一个定长向量表示**，供下游模型使用。对于输入的词序列，词元表示模型将每个词拆分为词表内的词元。例如，将单词“reborn”拆分为“re”和“born”。

模型随后查询每个词元的表示，将输入重新组成为词元表示序列。当下游模型需要计算一个单词或词组的表示时，可以将对应范围内的词元表示合成为需要的表示。因此，词元表示模型能够较好地解决自然语言处理系统中未登录词的问题。**词元分析（Tokenization）目标是将原始文本分割成由词元（Token）序列的过程。词元切分也是数据预处理中至关重要的一步。**

##### BPE

**字节对编码（Byte Pair Encoding，BPE）**模型是一种常见的子词词元模型。该模型所采用的词表包含最常见的单词以及高频出现的子词。在使用中，常见词通常本身位于 BPE 词表中，而罕见词通常能被分解为若干个包含在 BPE 词表中的词元，从而大幅度降低未登录词的比例。BPE 算法包括两个部分：（1）词元词表的确定；（2）全词切分为词元；（3）获得词元表示序列。计算过程下图所示。

<img src="..\..\img\llm-basic\BPE.png" alt="Image" style="zoom:67%;" />

在词元词表确定之后，对于输入词序列中未在词表中的全词进行切分，BPE 算法对词表中的词元按**从长到短**的顺序进行遍历，用每一个词元和当前序列中的全词或未完全切分为词元的部分进行匹配，将其切分为该词元和剩余部分的序列。

例如，对于单词“lowest\</w>”，首先通过匹配词元“est\</w>”将其切分为“low”, “est\</w>”的序列，再通过匹配词元“low”，确定其最终切分结果为“low”, “est\</w>”的序列。通过这样的过程，BPE 尽量将词序列中的词切分成已知的词元。

此外，字节级（Byte-level）BPE 通过将字节视为合并的基本符号，用来改善多语言语料库（例如包含非 ASCII 字符的文本）的分词质量。**GPT-2、BART 和 LLaMA** 等大语言模型都采用了这种分词方法。

原始 LLaMA 的词表大小是 32K，并且主要根据英文进行训练，因此，很多汉字都没有直接出现在词表中，需要字节来支持所有的中文字符，由 2 个或者 3 个 Byte Token 才能拼成一个完整的汉字。

###### BPE 输出

对于使用了字节对编码的大语言模型，其输出序列也是词元序列。对于原始输出，根据终结符 \</w> 的位置确定每个单词的范围，合并范围内的词元，将输出重新组合为词序列，作为最终的结果。

```python
# BPE
from transformers import AutoTokenizer
from collections import defaultdict

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# 使用 GPT-2 tokenizer 将输入分解为单词:
tokenizer = AutoTokenizer.from_pretrained("gpt2")

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# 计算基础词典, 这里使用语料库中的所有字符:
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# 增加特殊 Token 在字典的开头，GPT-2 中仅有一个特殊 Token``<|endoftext|>''表示文本结束
vocab = ["<|endoftext|>"] + alphabet.copy()

# 将单词切分为字符
splits = {word: [c for c in word] for word in word_freqs.keys()}

#compute_pair_freqs 函数用于计算字典中所有词元对的频率
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

#merge_pair 函数用于合并词元对
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

# 迭代训练，每次选取得分最高词元对进行合并，直到字典大小达到设置目标为止:
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

# 训练完成后，tokenize 函数用于给定文本进行词元切分
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])

tokenize("This is not a token.")
```



##### WordPiece 分词

WordPiece也是一种常见的词元分析算法，**最初应用于语音搜索系统**。此后，该算法**做为 BERT 的分词器**。

WordPiece 与 BPE 有非常相似的思想，都是通过迭代地合并连续的词元，但在合并的选择标准上略有不同。

为了进行合并，**WordPiece 需要首先训练一个语言模型**，并用该语言模型对所有可能的词元对进行评分。在每次合并时，选择使得训练数据**似然概率增加最多的词元对**。由于 Google 并没有发布其 WordPiece 算法的官方实现，HuggingFace 在其在线 NLP 课程中提供了一种更直观的选择度量方法：**一个词元对的评分是根据训练语料库中两个词元的共现计数除以它们各自的出现计数的乘积。**

##### Unigram 分词

Unigram 词元分析是另外一种应用于大语言模型的词元分析方法，T5 和 mBART 采用该方法构建词元分析器。不同于 BPE 和 WordPiece，Unigram 词元分析**从一个足够大的可能词元集合开始，然后迭代地从当前列表中删除词元，直到达到预期的词汇表大小为止。**

基于训练好的 Unigram 语言模型，使用从当前词汇表中删除某个字词后，训练语料库**似然性的增加量**作为选择标准。为了估计一元语言（Unigram）模型，采用了**期望最大化（Expectation–Maximization，EM）**算法：每次迭代中，首先根据旧的语言模型找到当前最佳的单词切分方式，然后重新估计一元语言单元概率以更新语言模型。

在这个过程中，使用**动态规划算法（如维特比算法）**来高效地找到给定语言模型时单词的最佳分解方式。

#### 冗余去除

大语言模型训练语料库中的重复数据，会降低语言模型的多样性，并可能导致训练过程不稳定，从而影响模型性能。因此，需要对预训练语料库中的重复进行处理，去除其中的冗余部分，这对于改善语言模型的训练具有重要的作用。

**文本冗余发现（Text Duplicate Detection）也称为文本重复检测**，是自然语言处理和信息检索中的基础任务之一，其目标是发现不同粒度上的文本重复，包括句子、段落以及文档等不同级别。冗余去除就是在不同的粒度上进行去除重复内容，包括句子、文档和数据集等粒度的重复。

##### 句子级别

包含重复单词或短语的句子很可能造成语言建模中引入重复的模式。这对语言模型来说会产生非常严重的影响，使得模型在预测时容易陷入重复循环（Repetition Loops）

例如，使用 GPT-2 模型，对于给定的上下文：

> In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

如果使用**束搜索（Beam Search）**，在设置 b = 32 时，模型就会产生如下输出，进入了重复循环模式：

> The study, published in the Proceedings of the National Academy of Sciences of the United States of America (PNAS), was conducted by researchers from the Universidad Nacional Autónoma de México (UNAM) and the Universidad Nacional Autónoma de México (UNAM/Universidad Nacional Autónoma de México/Universidad Nacional Autónoma de México/Universidad Nacional Autónoma de México/Universidad Nacional Autónoma de ...

由于重复循环对于语言模型生成的文本质量有非常大的影响，因此在预训练语料中需要删除这些包含大量重复单词或者短语的句子。

<img src="..\..\img\llm-basic\refinedweb.png" alt="image-20240307142026306" style="zoom:50%;" />

##### 文档级别

**在文档级别上，大部分大语言模型都是依靠文档之间的表面特征相似度（例如 重叠比例）进行检测并删除重复文档。**

**LLaMA 采用 CCNet 的处理模式**，首先将文档拆分为段落，并把所有字符转换为小写字符、将数字替换为占位符，以及删除所有 Unicode 标点符号和重音符号来对每个段落进行规范化处理。然后，使用 SHA-1 方法为每个段落计算一个哈希码（Hash Code），并使用前 64 位数字作为键。最后，利用每个段落的键进行重复判断。

RefinedWeb 首先去除掉页面中菜单、标题、页脚、广告等内容，仅抽取页面中的主要内容。在此基础上，在文档级别进行过滤，采用与文献类似的方法，使用 $n-gram$ 重叠程度来衡量句子、段落以及文档的相似度。如果重复程度超过预先设定的阈值，则会过滤掉重复段落或文档。

##### 数据集层面

数据集层面也可能存在一定数量的重复情况，比如很多大语言模型预训练集合都会包含 GitHub、Wikipedia、C4 等数据集。还需要特别注意的是，**预训练语料中混入测试语料，从而造成数据集污染的情况。**



#### 低质过滤

如何**从收集到的数据中删除低质量数据成为大语言模型训练中的重要步骤。**大语言模型训练中所使用的低质量数据过滤方法可以大致分为两类：**基于分类器的方法和基于启发式的方法。**

##### 基于分类器的方法

**基于分类器的方法目标是训练文本质量判断模型，并利用该模型识别并过滤低质量数据。**

GPT3、PALM 以及 GLam 模型在训练数据构造时都使用了基于分类器的方法。

文献采用了**基于特征哈希的线性分类器（Feature Hash Based Linear Classifier）**，可以非常高效地完成文本质量判断。该分类器使用一组精选文本（维基百科、书籍和一些选定的网站）进行训练，目标是将与训练数据类似的网页给定较高分数。利用这个分类器可以评估网页的内容质量。在实际应用中，还可以通过**使用 Pareto 分布对网页进行采样**，根据其得分选择合适的阈值，从而选定合适的数据集合。但是，一些研究也发现，**基于分类器的方法可能会删除包含方言或者口语的高质量文本，从而损失一定的多样性。**

##### 基于启发式的方法

基于启发式的方法则**通过一组精心设计的规则来消除低质量文本，**BLOOM 和 Gopher 采用了基于启发式的方法。

这些启发式规则主要包括：

- **语言过滤：**如果一个大语言模型仅关注一种或者几种语言，那么就可以大幅度的过滤掉数据中其他语言的文本。
- **指标过滤：**利用评测指标也可以过滤低质量文本。例如，可以使用语言模型对于给定文本的困惑度（Perplexity）进行计算，利用该值可以过滤掉非自然的句子。
- **统计特征过滤：**针对文本内容可以计算包括标点符号分布、符号字比（Symbol-to-Word Ratio）、句子长度等等在内的统计特征，利用这些特征过滤低质量数据。
- **关键词过滤：**根据特定的关键词集，可以识别和删除文本中的噪声或无用元素，例如，HTML 标签、超链接以及冒犯性词语等。

在大语言模型出现之前，在自然语言处理领域已经开展了很多**文章质量判断（Text Quality Evaluation）**相关研究，主要应用于**搜索引擎、社会媒体、推荐系统、广告排序以及作文评分**等任务中。

在搜索和推荐系统中，结果的内容质量是影响用户体验的的重要因素之一，因此，此前很多工作都是针对**用户生成内容（User-Generated Content，UGC）**质量进行判断。

自动作文评分也是文章质量判断领域的一个重要子任务，自 1998 年文献提出了**使用贝叶斯分类器进行作文评分预测以来，基于 SVM、CNN-RNN、BERT 等方法的作文评分算法也相继提出，并取得了较大的进展。**

这些方法也都可以应用于大语言模型预训练数据过滤中。**但是由于预训练数据量非常大，并且对于质量判断的准确率并不要求非常高，因此一些基于深度学习以及基于预训练的方法还没有应用于低质过滤过滤中。**

#### 隐私消除

由于绝大多数预训练数据源于互联网，因此不可避免地会包含涉及**敏感或个人信息（Personally Identifiable Information，PII）**的用户生成内容，这可能会增加隐私泄露的风险。如下图所示，输入前缀词“East Stroudsburg Stroudsburg”，语言模型在此基础上补全了姓名、电子邮件地址、电话 号码、传真号码以及实际地址。这些信息都是模型从预训练语料中学习得到的。因此，有非常必要从预训练语料库中删除包含个人身份信息的内容。

<img src="..\..\img\llm-basic\pii.png" alt="Image" style="zoom:50%;" />

删除隐私数据最直接的方法是采用基于规则的算法，BigScience ROOTS Corpus 构建过程中就是采用了基于**命名实体识别**的方法，利用命名实体识别算法检测姓名、地址和电话号码等个人信息内容并进行删除或者替换。该方法使用了基于 Transformer 的模型，并结合机器翻译技术，可以处理超过 100 种语言的文本，消除其中的隐私信息。该算法被集成在 **muliwai** 类库中。