## LLM主流结构和训练目标

预训练模型分为三类，分别是：**自编码**、**自回归**、**编码解码**。三种训练模型分别在前面三种任务中表现良好。

**自回归**（比如 GPT）：从左往右学习的模型，根据句子中前面的单词，预测下一个单词。例如，通过“LM is a typical task in natural language ____”预测单词“processing”。在长文本的生成能力很强，缺点就是**单向的注意力机制**在 NLU 任务中，不能完全捕捉 token 的内在联系。

**自编码**（比如 BERT）：通过覆盖句中的单词，或者对句子做结构调整，让模型复原单词和词序，从而调节网络参数。例如，可以把 BERT 看成一种自编码器，它通过 Mask 改变了部分 Token，然后试图通过其上下文的其它Token 来恢复这些被 Mask 的 Token。**自编码在语言理解相关的文本表示效果很好。缺点是不能直接用于文本生成。**

**编码解码**（比如T5）：编码器使用双向注意力，解码器使用单向注意力，并且有交叉注意力连接两者，在有条件生成任务（seq-seq）中表现良好(文本摘要，回答生成)。

这些训练框架都不足以在所有 NLP 中具有竞争力任务。以往的工作（T5）试图通过多任务学习统一不同的框架。然而，由于自编码和自回归的目标性质不同，一个简单的统一的优点不能完全继承这两个框架。

### 主流结构

目前LLM（Large Language Model）主流结构包括三种范式，分别为**Encoder-Decoder**、**Causal Decoder**、**Prefix Decoder**，如下图所示：

<img src="..\..\img\llm-basic\three-structures.png" alt="图片" style="zoom: 80%;" />

- Encoder-Decoder
  结构特点：输入双向注意力，输出单向注意力
  代表模型：**T5、Flan-T5、BART**
- Causal Decoder
  结构特点：从左到右的单向注意力
  代表模型：**LLaMA1/2系列**
- Prefix Decoder
  结构特点：输入双向注意力，输出单向注意力
  代表模型：**ChatGLM**、ChatGLM2、U-PaLM

### 结构对比

三种结构主要区别在于Attention Mask不同，如下图所示

<img src="..\..\img\llm-basic\three-masks.png" alt="图片" style="zoom: 67%;" />

- Encoder-Decoder
  特点：**在输入上采用双向注意力**，对问题的编码理解更充分;
  缺点：**在长文本生成任务上效果差，训练效率低**；
  适用任务：在偏理解的 NLP 任务上效果好。
- Causal Decoder
  特点：**自回归语言模型**，预训练和下游应用是完全一致的，**严格遵守只有后面的token才能看到前面的token的规则**；
  优点：训练效率高，zero-shot 能力更强，具有涌现能力；
  适用任务：文本生成任务效果好
- Prefix Decoder
  特点：**Prefix部分的token互相能看到**，属于Causal Decoder 和 Encoder-Decoder 的折中；
  缺点：训练效率低。

### 训练目标

#### 语言模型

根据已有词预测下一个词，即Next Token Prediction，是目前大模型所采用的最主流训练方式，训练目标为最大似然函数：
$$
L_{LM}(x) = \sum^n_{i=1}logP(x_i|x_{<i})
$$


训练效率：**Prefix Decoder < Causal Decoder**

Causal Decoder 结构会在**所有token上计算损失**，而Prefix Decoder只会在输出上计算损失。

#### 去噪自编码器

随机替换掉一些文本段，训练语言模型去恢复被打乱的文本段，即完形填空，训练目标函数为:
$$
L_{DAE}(x)=logP(\hat{x}|x_{/\hat{x}})
$$


去噪自编码器的实现难度更高，采用去噪自编码器作为训练目标的任务有GLM-130B、T5等。



