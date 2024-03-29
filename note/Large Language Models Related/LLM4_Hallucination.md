## 幻觉（Hallucination）

不遵循原文（Faithfulness）或者不符合事实（Factualness）

在传统任务里，幻觉大都是指的是Faithfulness：

- Intrinsic Hallucination（信息冲突）
  LMs在生成回复时，与输入信息产生了冲突，例如摘要问题里，abstract和document的信息不一致。
- Extrinsic Hallucination（无中生有）
  LMs在生成回复时，输出一些并没有体现在输入中的额外信息，比如邮箱地址、电话号码、住址，并且难以验证其真假。（PS: 按照此定义，Extrinsic Hallucination有可能是真的信息，只是需要外部信息源进行认证）

**而面向LLMs，我们通常考虑的幻觉则是Factualness。**

因为我们应用LLM的形式是open-domain Chat，而不是局限于特定任务，所以数据源可以看做任意的世界知识。LLMs如果生成了不在input source里的额外信息，但是符合事实的，这种情况也可能是对我们有帮助的。

LLM幻觉（Hallucination）经常表现为一本正经的胡说八道：看似流畅自然的表述，实则不符合事实或者是错误的。

LLM幻觉主要可以分为两类：即**内在幻觉**和**外在幻觉**。

- 内在幻觉
  生成的内容与源内容相矛盾。
- 外部幻觉
  生成的内容不能从源内容中得到验证，既不受源内容支持也不受其反驳。

LLMs的幻觉可能会产生如传播错误信息或侵犯隐私等严重后果。eg: 在医疗应用中，对患者生成的报告如果存在幻觉可能导致错误诊断甚至影响生命安全。

幻觉影响了模型的可靠性和可信度，因此需要解决LLM的幻觉问题。

### 为什么LLM会产生幻觉？

- 大模型缺乏相关的知识，或者存储的知识是错的。这个问题主要是由于预训练数据决定的。
- 大模型高估了自己的能力，他可能不知道问题的边界，编造内容回复。
- 对齐问题，这里主要是说我们通过指令精调，模型可以基于我们的instruction做出回复，但是调教出来的大模型可能会**迎合我们的instruction**，也有可能在某个问题下，他不具备这方面的知识，而问题又必须让他回答，他就只能生成一个一本正经的胡说八道内容。
- 在decoder-only的结构下，生成策略我们是每次**生成一个token**，早期如果就错了，那么后期大模型只能将错就错，不会纠正之前的错误，这也就产生了**幻觉滚雪球**的现象。还有像top-k，top-p的采样策略也会有影响。（top-k是每次解码时选择k个最大的概率，随机采样；top-p，是选定一个概率阈值，如chatgpt中默认的是0.95,意思是我们选择词表概率加和到大于等于0.95时随机采样。
- 最后，如GPT之类的生成模型，其实**只是学会了文本中词汇间的统计规律**，所以它们生成内容的准确性仍然是有限的。

### 如何度量幻觉

最有效可靠的方式当然是靠人来评估，但是人工评估的成本太高了。因此有了一些自动化评估的指标：

- 命名实体误
  命名实体（NEs）是“事实”描述的关键组成部分，我们可以利用**NE匹配**来计算生成文本与参考资料之间的一致性。直观上，如果一个模型生成了不在原始知识源中的NE，那么它可以被视为产生了幻觉（或者说，有事实上的错误）。
- 蕴含率
  该指标定义为被参考文本所蕴含的句子数量与生成输出中的总句子数量的比例。为了实现这一点，可以采用成熟的**蕴含/NLI模型**。
- 基于模型的评估
  应对复杂的句法和语义变化。
- 利用问答系统
  此方法的思路是，如果生成的文本在事实上与参考材料一致，那么对同一个问题，其答案应该与参考材料相似。具体而言，对于给定的生成文本，问题生成模型会创建一组问题-答案对。接下来，问答模型将使用原始的参考文本来回答这些问题，并计算所得答案的相似性。
- 利用信息提取系统
  此方法使用信息提取模型将知识简化为关系元组，例如<主体，关系，对象>。这些模型从生成的文本中提取此类元组，并与从原始材料中提取的元组进行比较。

### 如何缓解LLM幻觉

####  事实核心采样

《Factuality Enhanced Language Models for Open-Ended Text Generation》

在这种方法中，作者认为，采样的“随机性”在用于生成句子的后半部分时，对事实性的损害比在句子的开头更大。因为在句子的开始没有前文，所以只要它在语法和上下文上是正确的，LM就可以生成任何内容。然而，随着生成的进行，前提变得更为确定，只有更少的单词选择可以使句子成为事实。因此，他们引入了事实核心采样算法，该算法在生成每个句子时动态调整“核心”p。

#### 通过使用外部知识验证主动检测和减轻幻觉

《A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation》

作者发现

- 幻觉的生成是会传播的，比如一句话出现幻觉，后续生成的文本可能也会出现幻觉甚至更严重。这意味着，如果我们能够“主动”检测并减轻幻觉，那么我们也可以阻止其在后续生成的句子中的传播。
- logit输出值（输出词汇表上的概率分布）可以用来获取幻觉的信号。具体地说，我们计算了一个概率得分，并展示了当这个得分很低时，模型更容易产生幻觉。因此，它可以作为幻觉的一个信号，当得分很低时，可以对生成的内容进行信息验证。

基于这两个发现，作者提出了主动检测和减轻的方法。

在检测阶段，首先确定潜在幻觉的候选者，即生成句子的重要概念。然后，利用其logit输出值计算模型对它们的不确定性并检索相关知识。

在减轻阶段，使用检索到的知识作为证据修复幻觉句子。将修复的句子附加到输入（和之前生成的句子）上，并继续生成下一个句子。这个过程不仅减轻了检测到的幻觉，而且还阻止了其在后续生成的句子中的传播。

#### SelfCheckGPT

《SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models》

SelfCheckGPT的主要思想是：**如果模型真的掌握某个事实，那么多次生成的结果应该是相似的且事实一致的；相反，如果模型在胡扯，那么随机采样多次的结果会发散甚至矛盾。**

因此，他们从模型中采样多个response（比如通过变化温度参数）并测量不同response之间的信息一致性，以确定哪些声明是事实，哪些是幻觉。这种信息一致性可以使用各种方法计算，比如可以使用**神经方法计算语义等价（如BERTScore）或使用IE/QA-based方法。**