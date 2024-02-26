## Retrieval-Augmented Generation for Large Language Models: A Survey

[参考论文](https://arxiv.org/pdf/2312.10997.pdf)

**RAG(Retrieval-Augmented Generation)**， 检索增强生成，即从外部数据库获取额外信息辅助模型生成内容

<img src="..\..\img\rag\rag_instance.png" alt="image-20240226105726901" style="zoom: 50%;" />

**RAG**结合了检索（从大型外部数据库中提取信息）和生成（基于检索到的信息生成答案）两个步骤。RAG通过引入外部知识来源，来增强语言模型的回答能力。

**Fine-Tuning**是指在已有的预训练模型基础上，通过在特定数据集上进行额外的训练来优化模型。这个过程没有引入外部知识，而是调整模型以更好地适应特定任务。