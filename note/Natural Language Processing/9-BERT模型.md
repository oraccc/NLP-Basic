## BERT模型



### §9.1 BERT模型简介

**BERT的全称是Bidirectional Encoder Representation from Transformers**，是Google2018年提出的**预训练**模型，即双向Transformer的Encoder，因为Decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了 **Masked LM** 和 **Next Sentence Prediction** 两种方法分别捕捉词语和句子级别的representation。

---



### §9.2 从Word Embedding 到 BERT 模型的发展

