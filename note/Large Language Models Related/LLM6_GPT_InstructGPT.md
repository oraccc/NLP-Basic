## GPT1



GPT 是 Generative Pre-Training 的缩写，即**生成式预训练**，来自 OpenAI 的论文 Improving Language Understanding by Generative Pre-Training。Generative Pre-Training 包含两大方面的含义：

- Pre-Training：指的是大规模自监督预训练，即在大规模没有标签的文本数据上，自监督的完成模型训练，这样就可以利用并发挥出海量无标签文本数据的价值了；
- Generative：自监督预训练是通过Next Token Prediction的方式实现的，即是一种生成式的训练方式。

因此GPT的基本思想是：**先在大规模没有标签的数据集上训练一个预训练模型，即generative pre-training的过程；再在子任务小规模有标签的数据集上训练一个微调模型，即discriminative fine-tuning的过程。**

初代GPT选择使用 Transformer 结构，是因为在 NLP 的相关任务中，Transformer 学到的 features 更稳健。与循环神经网络等其他模型相比，Transformer 提供了更结构化的长期记忆，这有利于文本中的长期依赖关系的处理，从而更好的抽取句子层面和段落层面的语义信息。

GPT 利用了任务相关的输入这一表示，将结构化文本输入作为一个连续的 token 序列进行处理。

经典的 Transformer Decoder Block 包含3个子层, 分别是 Masked Multi-Head Attention 层, Encoder-Decoder Attention 层, 以及最后的一个 Feed Forward Network（FFN）全连接层。

**GPT是一个 Decoder-Only 的结构，他根本就没有编码器，自然无需从编码器中获得 Key 和 Value !** 因此，在Decoder-Only 的魔改 Transformer 中，我们往往会取消第二个 Encoder-decoder Attention 子层, 只保留Masked Multi-Head Attention 层, 和 Feed Forward Network 层。

GPT模型结构如下图所示：

<img src="..\..\img\llm-basic\gpt.png" alt="Image" style="zoom:50%;" />

- Embedding：词嵌入+位置编码（先前文章已讲过）；
- 带掩码的多头自注意力机制，让当前时间步和后续时间步的信息的点积在被softmax后都为零（先前文章已讲过）；
- 输出的向量输入给一个全连接层，而后再输出给下一个Decoder；
- GPT有**12**个Decoder，经过他们后最终的向量经过一个logit的linear层、一个softmax层，就输出了一个下一个词的概率分布函数；
- 输出的词会作为下一个词的输入。

### 自监督预训练

GPT 的自监督预训练是基于语言模型进行训练的。假设有一段文本，把每个词计作 $u_i$，则形成一个无标签的序列 $U={u_1,...,u_n}$，GPT 使用标准的语言模型目标函数来最大化下面的似然函数：
$$
L_1(U)=\sum_{i}logP(u_i|u_{i-k},...,u_{i-1};M)
$$
具体来说是要预测每个词$u_i$的概率$P$，这个概率是基于它前面$u_{i-k}$到$u_{i-1}$个词，以及模型$M$。这里的$k$表示上文的窗口大小，理论上来讲$k$取的越大，模型所能获取的上文信息越充足，模型的能力越强

模型对输入进行特征嵌入得到 transformer 第一层的输入，再经过多层 transformer （GPT为12层）特征编码，使用最后一层的输出即可得到当前预测的概率分布

### 有监督微调

<img src="..\..\img\llm-basic\gpt-finetuning.png" alt="image-20240305210718921" style="zoom: 67%;" />

### 任务相关的输入和输出变换

<img src="..\..\img\llm-basic\gpt-tasks.png" alt="Image" style="zoom:67%;" />

对于GPT处理的4个不同任务，这些任务有的只有一个输入，有的则有多组形式的输入。对于不同的输入，GPT有不同的处理方式，具体方式如下：

- **分类任务：**将起始和终止token加入到原始序列两端，输入transformer中得到特征向量，最后经过一个全连接得到预测的概率分布；
- **自然语言推理：**将前提（premise）和假设（hypothesis）通过分隔符（Delimiter）隔开，两端加上起始和终止token。再依次通过transformer和全连接得到预测结果；
- **语义相似度：**输入的两个句子，正向和反向各拼接一次，然后分别输入给transformer，得到的特征向量拼接后再送给全连接得到预测结果；
- **问答和常识推理：**将 $n$ 个选项的问题抽象化为 $n$ 个二分类问题，即每个选项分别和内容进行拼接，然后各送入transformer和全连接中，最后选择置信度最高的作为预测结果。

> 通过这样的方法，这四个自任务就都变成了序列+标注的形式。尽管各自的形式还是稍微有一些不一样，但不管输入形式如何、输出构造如何，中间的Transformer他是不会变的。不管怎样，我们都不去改图中的Transformer的模型结构，这是GPT和之前工作的区别，也是GPT这篇文章的一个核心卖点。

### GPT 与 BERT

#### GPT和BERT的区别

GPT使用的Transformer的Decoder层（目标函数为标准的语言模型，每次只能看到当前词之前的词，需要用到Decoder中的Masked attention）。

BERT使用的Transformer的Encoder层（目标函数为带[Mask]的语言模型，通过上下文的词预测当前词，对应Encoder）。

#### 为什么初代GPT的性能比BERT差？

GPT预训练时的任务更难（BERT的base就是为了和GPT对比，参数设定几乎一样）

BERT预训练用的数据集大小几乎是GPT的四倍。



## GPT2



基于GPT-1和BERT的工作，发现GPT-1这种上下文生成应用面更广。GPT-2的目的就是做这个事情，模型更大，数据更多，效果是否能干掉BERT。

于是作者收集了一个更大的数据集WebText，百万网页数据，同时将模型从1亿参数（110M）变成了15亿参数（1.5B）。但问题是，数据集上去了，模型规模上去了，效果真的能有明显的优势吗？于是作者想到了证明的路子：**Zero-Shot，也是 GPT-2 的核心观点。**

### GPT2 核心思想

<img src="..\..\img\llm-basic\gpt2-key.png" alt="image-20240305211240592" style="zoom:67%;" />

基于上面的思想，作者认为，当一个语言模型的容量足够大时，它就足以覆盖所有的有监督任务，也就是说**所有的有监督学习都是无监督语言模型的一个子集**。

例如当模型训练完“Micheal Jordan is the best basketball player in the history”语料的语言模型之后，便也学会了(question：“who is the best basketball player in the history ?”，answer:“Micheal Jordan”)的Q&A任务。

综上，GPT-2的核心思想概括为：**任何有监督任务都是语言模型的一个子集，当模型的容量非常大且数据量足够丰富时，仅仅靠训练语言模型的学习便可以完成其他有监督学习的任务。**

那么有了美好的想法，怎样实现呢？那就**是在大数据和大模型的加持下，由GPT-1的Pre-Training + Fine-Tuning，改成了GPT-2的Pre-Training + Prompt Predict （Zero-Shot Learning）。**

### 自监督预训练

训练方式同GPT-1一样，只是数据量变大，模型参数变大，模型结构只是做了几个地方的调整，这些调整更多的是被当作训练时的 trick，而不作为 GPT-2 的创新

### Zero-Shot Predict

下游任务转向做zero-shot而放弃微调（fine-tuning），相较于GPT-1，出现一个新的问题：样本输入的构建不能保持GPT-1的形态，因为模型没有机会学习[Start]，[Delim]，[Extract]这些特殊token。因此，**GPT-2使用一种新的输入形态：增加文本提示，后来被称为prompt**（不是GPT-2第一个提出，他使用的是18年被人提出的方案）。

#### Multitask Learning

GPT-2的论文名称叫《Language Models are Unsupervised Multitask Learners》，Unsupervised 好理解，因为GPT本来就是一个Unsupervised的任务，那么为什么要Multitask Learning，以及什么是Multitask Learning？

现在的语言模型泛化能力比较差，在一个训练集、一个训练任务上训练出来的参数很难直接用到下一个模型里。因此，目前大多数模型都是Narrow Expert，而不是Competent Generalists。**OpenAI希望朝着能够执行许多任务的更通用的系统发展——最终不需要为每个任务手动创建和标记训练数据集。**

**多任务学习的定义：**
多任务学习（Multi-Task Learning, MTL）是一种机器学习方法，它可以通过同时学习多个相关的任务来提高模型的性能和泛化能力。与单任务学习只针对单个任务进行模型训练不同，多任务学习通过共享模型的部分参数来同时学习多个任务，从而可以更有效地利用数据，提高模型的预测能力和效率。

如何做到多任务学习呢？

**把所有的任务都归结为上下文的问题回答。**具体的，应该满足如下的条件：

1. 必须对所有任务一视同仁，也就是**喂给模型的数据不能包含这条数据具体是哪个任务**，不能告诉模型这条数据是要做NMT，另一条数据要做情感分类。
2. 模型也不能包含针对不同任务的特殊模块。给模型输入一段词序列，**模型必须学会自己辨别这段词序列要求的不同的任务类型**，并且进行内部切换，执行不同的操作。
3. 模型还应具备执行训练任务之外的任务的能力，即 zero shot learning。

Multitask Question Answering Network, MQAN这篇文章中提出了一个新的在**没有任何特定任务模块或参数**的情况下**联合学习**decaNLP的**所有任务**。把**各种下游子任务都转换为QA任务**，并且让模型通过我们告诉他的自然语言（**Prompt**去自动执行不同的操作，从而完成任务的想法也是GPT-2的关键。这就是为什么提出GPT-2的论文的标题叫《Language Models are Unsupervised Multitask Learners》)。

#### Zero-Shot Learning

GPT-2 最大的改变是抛弃了前面“无监督预训练+有监督微调”的模式，而是开创性地引入了 Zero-shot 的技术，即预训练结束后，不需要改变大模型参数即可让它完成各种各样的任务。

**Zero-shot的含义：**我们用预训练模型做下游任务时，不需要任何额外的标注信息，也不去改模型参数。

#### 从 fine-tune 到 zero-shot

GPT-1中，我们的模型在自然语言上进行预训练，到了给下游任务做微调的时候，我们是引入了很多模型之前从来没有见过的特殊符号，这个符号是针对具体的任务专门设计的，即给GPT的输入进行了特殊的构造，加入了开始符、结束符、分割符*这些符号，模型要在微调的过程中慢慢认识。

如果想要做zero-short，即不做任何额外的下游任务训练的话，就没办法让模型去临时学习这些针对特定任务的构造了。因此，我们在构造下游任务的输入的时候，就不能引入特殊的符号，而是要让整个下游任务的输入和之前在预训练的时候看到的文本形式一样。**即要使得输入的形式应该更像自然语言。**

不论是 GPT-1 还是 BERT，NLP 任务中比较主流的 pre-train + fine-tuning 始终还是**需要一定量的下游任务有监督数据去进行额外的训练，在模型层面也需要额外的模块去进行预测，仍然存在较多人工干预的成本**。GPT-2 想彻底解决这个问题，**通过 zero-shot，在迁移到其他任务上的时候不需要额外的标注数据，也不需要额外的模型训练。**

在 GPT-1 中，下游任务需要对不同任务的输入序列进行改造，在序列中加入了开始符、分隔符和结束符之类的特殊标识符，但是在 zero-shot 前提下，我们无法根据不同的下游任务去添加这些标识符，**因为不进行额外的微调训练，模型在预测的时候根本不认识这些特殊标记**。所以**在 zero-shot 的设定下，不同任务的输入序列应该与训练时见到的文本长得一样，也就是以自然语言的形式去作为输入**，例如下面两个任务的输入序列是这样改造的：

> 机器翻译任务：translate to french, { english text }, { french text }
> 阅读理解任务：answer the question, { document }, { question }, { answer }

为什么上述输入序列的改造是有效的？或者说为什么 zero-shot 是有效的？这里引用原文的一句话：

> Our approach motivates building as large and diverse a dataset as possible in order to collect natural language demonstrations of tasks in as varied of domains and contexts as possible.

大概意思是，**从一个尽可能大且多样化的数据集中一定能收集到不同领域不同任务相关的自然语言描述示例**。所以再次印证 GPT-2 的核心思想就是，**当模型的容量非常大且数据量足够丰富时，仅仅靠语言模型的学习便可以完成其他有监督学习的任务，不需要在下游任务微调。**

#### 从 zero-shot 到 prompting

既然输入的形式也要更像自然语言，那么就应该让模型通过我们的自然语言，去知道现在要去执行什么任务。

要如何做：实现 Zero-shot learning 的前提就是，我们得能够做到不需要针对下游的任务，不给模型的输入结构做专门的设计；而是只需要给模型指示，也就是提示词（Prompt）就好了。

当数据量足够大、模型能力足够强的时候，语言模型会学会推测并执行用自然语言提出的任务，因为这样可以更好的实现下一词预测

### 模型结构

在模型结构方面，整个 GPT-2 的模型框架与 GPT-1 相同，只是做了几个地方的调整，这些调整更多的是被当作训练时的 trick，而不作为 GPT-2 的创新，具体为以下几点：

1. 后置层归一化**（ post-norm ）**改为前置层归一化**（ pre-norm ）**;
2. 在模型最后一个自注意力层之后，额外增加一个层归一化;
3. 调整参数的初始化方式，按残差层个数进行缩放，缩放比例为 1 : sqrt(n);
4. 输入序列的最大长度从 512 扩充到 1024;
5. 模型层数扩大，从 GPT-1 的 12 层最大扩大到 48 层，参数量也从 1 亿扩大到 15 亿。

其中，关于 post-norm 和 pre-norm 可以参考《Learning Deep Transformer Models for Machine Translation》。两者的主要区别在于，post-norm 将 transformer 中每一个 block 的层归一化放在了残差层之后，而 pre-norm 将层归一化放在了每个 block 的输入位置，如下图所示：

<img src="..\..\img\llm-basic\post-pre-norm.png" alt="Image" style="zoom:67%;" />

GPT-2 进行上述模型调整的主要原因在于，**随着模型层数不断增加，梯度消失和梯度爆炸的风险越来越大，Pre-Norm 能够减少预训练过程中各层之间的方差变化，使梯度更加稳定。**

GPT-2 在一些任务上与有监督微调的方法相比还是有一些差距的，这可能也是 GPT-2 在当时影响力没有那么大的一个原因。但 **GPT-2 在较多任务上对比无监督算法取得了一定的提升，证明了 zero-shot 的能力，这也成为了GPT系列的破晓晨光。**



## GPT3

GPT-1 模型指出，如果用 Transformer 的解码器和大量的无标签样本去预训练一个语言模型，然后在子任务上提供少量的标注样本做 **Fine-Tune**，就可以很大的提高模型的性能。

GPT-2 则是更往前走了一步，说在子任务上不去提供任何相关的训练样本，而是直接用足够大的预训练模型去理解自然语言表达的要求，并基于此做预测，因此主推了 **Zero-Shot**，虽然具有较高的创新度和吸引力，但其效果上还是不够惊艳，所以在业界也并没有取得比较大的影响力。

**为了解决效果上的问题，时隔一年多，GPT-3 以势不可挡之势卷土重来。**GPT-3 不再去追求那种极致的不需要任何样本就可以表现很好的模型，而是考虑像人类的学习方式那样，仅仅使用极少数样本就可以掌握某一个任务，因此就引出了 GPT-3 标题 Language Models are **Few-Shot** Learners。

**GPT-3 中的 few-shot learning，只是在预测是时候给几个例子，并不微调网络。**GPT-2用 zero-shot 去讲了 multitask Learning 的故事，GPT-3 使用 meta-learning 和 in-context learning 去讲故事。

### 自监督预训练

训练方式同 GPT-1 和 GPT-2 一样，只是数据量和模型参数都得到了巨幅提升，网络结构也做了一些优化

#### In-context learning

**In-context learning 是 GPT-3 运用的一个重要概念，本质上是属于 few-shot learning，只不过这里的 few-shot 只是在预测是时候给几个例子，并不微调网络，即不会再更新模型权重。**

要理解 in-context learning，我们需要先理解 meta-learning（元学习）。对于一个少样本的任务来说，模型的初始化值非常重要，从一个好的初始化值作为起点，模型能够尽快收敛，使得到的结果非常快的逼近全局最优解。Meta-learning 的核心思想在于**通过少量的数据寻找一个合适的初始化范围**，使得模型能够在有限的数据集上快速拟合，并获得不错的效果。

我们使用MAML（Model-Agnostic Meta-Learning）算法来理解一下 Meta-learning。正常的监督学习是将一个批次的数据打包成一个batch进行学习，但是元学习是将一个个任务打包成batch，每个batch分为支持集（support set）和质询集（query set），类似于学习任务中的训练集和测试集。

对一个网络模型 ，其参数表示为 ，它的初始化值被叫做meta-initialization。MAML的目标则是学习一组meta-initialization，能够快速应用到其它任务中。MAML的迭代涉及两次参数更新，分别是**内循环**（inner loop）和**外循环**（outer loop）。内循环是根据任务标签快速的对具体的任务进行学习和适应，而外学习则是对meta-initialization进行更新。直观的理解，我用一组meta-initialization去学习多个任务，如果每个任务都学得比较好，则说明这组meta-initialization是一个不错的初始化值，否则我们就去对这组值进行更新。

**GPT-3 中据介绍的 in-context learning（情境学习）就是元学习的内循环，而基于语言模型的SGD则是外循环**

<img src="..\..\img\llm-basic\gpt3-meta.png" alt="Image" style="zoom:67%;" />

GPT-3 的 few-shot learning 是不会做梯度下降，它是怎么做的？

**只做预测，不做训练。**我们希望 Transformer 在做前向推理的时候，能够通过注意力机制，从我们给他的输入之中抽取出有用的信息，从而进行预测任务，而预测出来的结果其实也就是我们的任务指示了。**这就是上下文学习（In context learning）。**

GPT-3 在下游任务的评估与预测时，提供了三种不同的方法：

- Zero-shot：仅使用当前任务的自然语言描述，不进行任何梯度更新；
- One-shot：当前任务的自然语言描述，加上一个简单的输入输出样例，不进行任何梯度更新；
- Few-shot：当前任务的自然语言描述，加上几个简单的输入输出样例，不进行任何梯度更新。

其中 Few-shot 也被称为 in-context learning，虽然它与 fine-tuning 一样都需要一些有监督标注数据，但是两者的区别是：

1. fine-tuning 基于标注数据对模型参数进行更新，而 in-context learning 使用标注数据时不做任何的梯度回传，模型参数不更新**【本质区别】**；
2. in-context learning 依赖的数据量（10～100）远远小于 fine-tuning 一般的数据量。

最终通过大量下游任务实验验证，Few-shot 效果最佳，One-shot 效果次之，Zero-shot 效果最差

当然这个模式也有**缺陷**：

1. GPT-3 的输入窗口长度是有限的，不可能无限的堆叠example的数量，即有限的输入窗口限制了我们利用海量数据的能力；
2. 每次做一次新的预测，模型都要从输入的中间抓取有用的信息。可是我们做不到把从上一个输入中抓取到的信息存起来，存在模型中，用到下一次输入里。

### 模型结构

GPT-3 沿用了GPT-2 的结构，但是在网络容量上做了很大的提升，并且使用了一个 **Sparse Transformer** 的架构，具体如下：

1. GPT-3采用了96层的多头transformer，头的个数为 96；
2. 词向量的长度是12,888；
3. 上下文划窗的窗口大小提升至2,048个token；
4. 使用了alternating dense和 locally banded sparse attention。

sparse attention 与传统 self-attention（称为 dense attention） 的区别在于：

- dense attention：每个 token 之间两两计算 attention，复杂度 O(n²)
- sparse attention：**每个 token 只与其他 token 的一个子集计算 attention，复杂度 O(n*logn)**

具体来说，sparse attention 除了相对距离不超过 k 以及相对距离为 k，2k，3k，... 的 token，其他所有 token 的注意力都设为 0，如下图所示：

<img src="..\..\img\llm-basic\sparse_attention.png" alt="Image" style="zoom:67%;" />

使用 sparse attention 的好处主要有以下两点：

1. 减少注意力层的计算复杂度，节约显存和耗时，从而能够处理更长的输入序列；
2. 具有“**局部紧密相关和远程稀疏相关**”的特性，对于距离较近的上下文关注更多，对于距离较远的上下文关注较少。

### 训练数据

GPT-3 使用了多个数据集，其中最大的是 CommonCrawl，原始未处理的数据达到了 45TB，其实在 GPT-2 的时候他们就有考虑使用这个数据集，但是后来还是觉得这个数据集太脏了所以没用，但是现在 GPT-3 的模型规模太大了，使得训练对数据量的需求也增加了很多，他们不得不重新考虑这个数据集。因此，他们必须在这个数据集上做一些额外的数据清洗工作来尽量保证数据的质量。

数据处理主要包括以下几个部分：

1. 使用高质量数据作为正例，训练分类算法，对 CommonCrawl 的所有文档做初步过滤
2. 利用公开的算法做文档去重，减少冗余数据
3. 加入已知的高质量数据集



## InstructGPT

GPT-3 虽然在各大 NLP 任务以及文本生成的能力上令人惊艳，但是它仍然还是会**生成一些带有偏见的，不真实的，有害的造成负面社会影响的信息，而且很多时候，他并不按人类喜欢的表达方式去说话。**

在这个背景下，OpenAI 提出了一个 **Alignment** 概念，意思是模型输出与人类真实意图对齐，符合人类偏好。因此，为了让模型输出与用户意图更加 “align”，就有了 InstructGPT 这个工作。

InstructGPT 提出了一个理想化语言模型的 3H 目标：

- **Helpful:** 能帮助用户解决问题
- **Honest:** 不能捏造事实，不能误导用户
- **Harmless:** 不能对用户或环境造成物理、精神、社会层面的伤害

从做研究的角度来讲，其实很多时候人们并不在意 “Alignment”问题，只要一个模型在评估的数据集上表现好，那基本就可以说是一个好模型。

关于 InstructGPT 的技术方案，原文分为了三个步骤：**有监督微调，奖励模型训练，强化学习训练**；实际上可以把它拆分成两种技术方案，一个是**有监督微调（SFT）**，一个是**基于人类反馈的强化学习（RLHF）**。

### SFT

本质上来说，**SFT 可以理解为人工标注了一批数据，然后去微调 GPT-3。**但是值得一提的是，这里标注的数据与 GPT-3 之前用来做下游任务使用的 few-shot 格式，有非常本质的区别。

**GPT-3 中的 few-shot 对于同一个下游任务，通常采用固定的任务描述方式，而且需要人去探索哪一种任务表述方式更好。**显然**这种模式与真实场景下用户的使用方式存在较大的 gap**，用户在向 GPT-3 提问时才不会采用某种固定的任务表述，而是随心所欲地以自己的说话习惯去表达某个需求。

InstructGPT 在 SFT 中标注的数据，正是为了消除这种模型预测与用户表达习惯之间的 gap。在标注过程中，他们**从 GPT-3 的用户真实请求中采样大量下游任务的描述，然后让标注人员对任务描述进行续写，从而得到该问题的高质量回答**。这里用户真实请求又被称为某个任务的指令（Instruct），即 InstructGPT 的核心思想“基于人类反馈的指令微调”。

### 基于人类反馈的强化学习（RLHF）

基于 SFT 得到的模型被用于接下来的 RLHF 做进一步的模型优化，流程如下

<img src="..\..\img\llm-basic\instruct_rlhf.png" alt="Image" style="zoom: 67%;" />

如上图所示，以摘要生成任务为例，详细展示了如何基于人类反馈进行强化学习，最终训练完成得到 InstructGPT 模型。主要分为三步：

1. **收集人类反馈：**使用初始化模型对一个样本生成多个不同摘要，人工对多个摘要按效果进行排序，得到一批排好序的摘要样本；
2. **训练奖励模型：**使用第1步得到的样本集，训练一个模型，该模型输入为一篇文章和对应的一个摘要，模型输出为该摘要的得分；
3. **训练策略模型：**使用初始化的策略模型生成一篇文章的摘要，然后使用奖励模型对该摘要打分，再使用打分值借助 PPO （Proximal Policy Optimization，近端策略优化）算法重新优化策略模型。

### 整体流程

InstructGPT 整体训练流程如下图三步所示：

<img src="..\..\img\llm-basic\instructgpt.png" alt="Image" style="zoom: 67%;" />

而 ChatGPT 的训练流程如下图所示：

<img src="..\..\img\llm-basic\chatgpt_train.png" alt="Image" style="zoom:67%;" />

总的来说，InstructGPT 相对于之前的 GPT 系列，有以下几点值得注意：

> 1. 解决 GPT-3 的输出与人类意图之间的 Align 问题；
> 2. 让具备丰富世界知识的大模型，学习“人类偏好”；
> 3. 标注人员明显感觉 InstructGPT 的输出比 GPT-3 的输出更好，更可靠；
> 4. InstructGPT 在真实性，丰富度上表现更好；
> 5. InstructGPT 对有害结果的生成控制的更好，但是对于“偏见”没有明显改善；
> 6. 基于指令微调后，在公开任务测试集上的表现仍然良好；
> 7. InstructGPT 有令人意外的泛化性，在缺乏人类指令数据的任务上也表现很好。

