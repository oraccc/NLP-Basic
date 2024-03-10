# Freq Q&A

## 1. Multi-head self-attention

#### PyTorch版实现

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V \\
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O \\
(head_i=Attention(QW^Q_i, KW^K_i, VW^V_i))
$$



```python
def attention(query, key, value, mask=None, dropout=None):
	'''
	缩放点积注意力
	query, key, value: [batch_size, (num_heads), seq_len, d_model]
	mask: [batch_size, (num_heads), 1, seq_len]
	'''
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	attn = scores.softmax(dim=-1)
	if dropout is not None:
    	attn = dropout(attn)
    # Output:  [batch_size, (num_heads), seq_len, d_model]
	return torch.matmul(p_attn, value), attn
```

```python
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
    	super(MultiHeadedAttention, self).__init__()
    	assert d_model % h == 0
    	# We assume d_v always equals d_k
    	self.d_k = d_model // h
    	self.h = h
        # 在多头注意力机制中，典型的实现方式是将查询向量、键向量和值向量分别通过线性变换映射到不同的子空间，然后再进行注意力计算。为了实现这一点，通常会使用多个线性层来分别处理这三个向量。在这个实现中，为了确保可以并行地处理查询、键和值，除了处理这三个向量外，还添加了一个额外的线性层用于最后的输出。
    	self.linears = clones(nn.Linear(d_model, d_model), 4)
    	self.attn = None
    	self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
        # query, key, value: [batch_size, seq_len, d_model]
    	if mask is not None:
      	# Same mask applied to all h heads.
      		mask = mask.unsqueeze(1)
    	num_batches = query.size(0)

    	# 1) Do all the linear projections in batch from d_model => h*d_k
    	query, key, value = [
			linear(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
      		for linear, x in zip(self.linears, (query, key, value))
    	]
        # query, key, value: [batch_size, num_heads, seq_len, d_model]

    	# 2) Apply attention on all the projected vectors in batch.
        # x: [batch_size, num_heads, seq_len, d_model]
    	x, self.attn = attention(
			query, key, value, mask=mask, dropout=self.dropout
    	)
        

		# 3) "Concat" using a view and apply a final linear.
    	x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.h * self.d_k)
        del query
    	del key
    	del value
        # Output: [batch_size, seq_len, d_model]
    	return self.linears[-1](x)
```



## Transformer其余结构

### Position Encoding

```python
import math
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model:int, max_seq_len:int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.d_model = d_model
        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        # 在位置编码张量的维度上添加一个维度，使其形状变为 (1, max_seq_len, d_model)，以便与输入张量相加。
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        # 从位置编码张量中提取与输入序列相对应的位置编码，并且将其从计算图中分离，使得它不会参与梯度计算
        pe = self.pe[:,:seq_len].detach().requires_grad_(False)
        # 通过广播机制实现的相加
        return self.dropout(x + pe)
```

### Decoder (Contains 6 Decoder Layer)

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/encoder-decoder.png" width="450" />

```python
class SublayerConnection(nn.Module):
  """
  A residual connection followed by a layer norm.
  """

  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    # Pre Norm
    return x + self.dropout(sublayer(self.norm(x)))
	# Post Norm
	return self.norm(x + self.dropout(sublayer(x)))
```

```python
class Decoder(nn.Module):
  "Generic N layer decoder with masking. (N=6)"

  def __init__(self, layer, N):
    super(Decoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = LayerNorm(layer.size)

  def forward(self, x, memory, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)
```

```python
class DecoderLayer(nn.Module):
  "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

  def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size, dropout), 3)

  def forward(self, x, memory, src_mask, tgt_mask):
    "Follow Figure 1 (right) for connections."
    m = memory
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
    x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
    return self.sublayer[2](x, self.feed_forward)
```

### Subsequent_Mask

```python
def subsequent_mask(size):
  "Mask out subsequent positions."
  attn_shape = (1, size, size)
  subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
      torch.uint8
  )
  return subsequent_mask == 0
```

### Full Model

```python
class EncoderDecoder(nn.Module):
  """
  A standard Encoder-Decoder architecture. Base for this and many
  other models.
  """

  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator

  def forward(self, src, tgt, src_mask, tgt_mask):
    "Take in and process masked src and target sequences."
    return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)

  def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```



```python
def make_model(
  src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
  "Helper: Construct a model from hyperparameters."
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout)
  model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
    Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
    nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
    Generator(d_model, tgt_vocab),
  )

  # This was important from their code.
  # Initialize parameters with Glorot / fan_avg.
  for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  return model
```

### Inference

```python
def inference_test():
  test_model = make_model(11, 11, 2)
  test_model.eval()
  src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
  src_mask = torch.ones(1, 1, 10)

  memory = test_model.encode(src, src_mask)
  ys = torch.zeros(1, 1).type_as(src)

  for i in range(9):
    out = test_model.decode(
      memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
    )
    prob = test_model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.data[0]
    ys = torch.cat(
      [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
    )

  print("Example Untrained Model Prediction:", ys)


def run_tests():
  for _ in range(10):
    inference_test()

show_example(run_tests)
```



## 2. Post-Norm和Pre-Norm

**Bert的Post-Norm**，是在Add操作后进行Norm操作，因此叫做Post-Norm

而Pre-Norm则是Norm之后再Add，所以叫Pre-Norm

<img src="..\..\img\llm-basic\post-pre-norm.png" alt="Image" style="zoom:67%;" />

Post-Norm由于是在残差之后进行归一化，因此归一化的效果更好，使得模型的**鲁棒性更强**。

而Pre-Norm由于并不是所有的参数都参与正则化，因此整体来说更**不容易发生梯度消失**的问题，模型训练的稳定性更强。Pre Norm结构无形地增加了模型的宽度而降低了模型的深度

因此，在Bert时代由于层数较浅，往往采用的是Post-Norm，而到了大模型时代，由于transformer的层数开始加深，为了训练稳定性开始使用Pre-Norm。

> **【梯度消失】**经常出现，产生的原因有：一是在**深层网络**中，二是采用了**不合适的损失函数**，比如sigmoid。当梯度消失发生时，接近于输出层的隐藏层由于其梯度相对正常，所以权值更新时也就相对正常，但是当越靠近输入层时，由于梯度消失现象，会导致靠近输入层的隐藏层权值更新缓慢或者更新停滞。这就导致在训练时，只等价于后面几层的浅层网络的学习。
>
> **【梯度爆炸】**一般出现在**深层网络**和**权值初始化值太大**的情况下。在深层神经网络或循环神经网络中，**误差的梯度可在更新中累积相乘**。如果网络层之间的**梯度值大于 1.0**，那么**重复相乘会导致梯度呈指数级增长**，梯度变的非常大，然后导致网络权重的大幅更新，并因此使网络变得不稳定。
>
> [Link](https://cloud.tencent.com/developer/article/1700046)



## 3. Layer Norm 与 Batch Norm

BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。**BN的具体做法就是对每一小批数据，在批这个方向上做归一化**。

* 对每一个batch进行操作，使得对于这一个batch中所有的输入数据，它们的每一个特征都是均值为0，方差为1的分布 

* 单纯把所有的输入限制为(0,1)分布也是不合理的，这样会降低数据的表达能力（第L层辛苦学到的东西，这里都暴力变成(0,1)分布了）。因此需要再加一个线性变换操作，让数据恢复其表达能力。这个线性变化中的两个参数 $\gamma$, $\beta$ 是需要模型去学习的。

LN整体做法类似于BN，不同的是LN不是在特征间进行标准化操作（横向操作），而是在整条数据间进行标准化操作**（纵向操作）**。它也是归一化数据的一种方式，不过**LN 是在每一个样本上计算均值和方差**，而不是BN那种在批方向计算均值和方差

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/normalization.png" width="650" />



## 4. ChatGLM

[link](https://www.zhihu.com/tardis/zm/art/637382548?source_id=1005)

目前预训练语言模型主要有三种类型：自回归模型、自编码模型和编码器-解码器模型。

**自回归模型**从左到右学习语言模型，适合于长文本生成和少样本学习，但不能捕捉上下文词之间的双向依赖关系。

**自编码模型**通过去噪目标学习双向上下文编码器，适合于自然语言理解任务，但不能直接用于文本生成。

**编码器-解码器模型**结合了双向注意力和单向注意力，适合于有条件的生成任务，如文本摘要和回复生成。

这三类语言模型各有优缺点，但没有一种框架能够在所有的自然语言处理任务中都表现出色。一些先前的工作尝试通过多任务学习的方式，将不同框架的目标结合起来，但由于自编码和自回归目标本质上的不同，简单的结合不能充分继承两者的优势。因此，清华大学提出了一种基于**自回归空白填充的通用语言模型（GLM）**，来解决这个挑战。GLM 通过添加**二维位置编码**和允许任意顺序预测空白区域，改进了空白填充预训练，在自然语言理解任务上超越了 BERT 和 T5。

GLM 从输入文本中随机挖掉一些连续的词语，然后训练模型按照一定的顺序逐个恢复这些词语。这种方法结合了自编码和自回归两种预训练方式的优点。GLM 还有两个改进点，一个是**打乱空白区域的预测顺序**，另一个是**使用二维位置编码**。

清华大学提出了一种基于自回归空白填充目标的通用预训练框架 GLM。GLM 将 NLU 任务转化为包含任务描述的完形填空问题，可以通过自回归生成的方式来回答。自回归空白填充目标是指在输入文本中随机挖去一些连续的文本片段，然后训练模型**按照任意顺序重建这些片段**。完形填空问题是指在输入文本中用一个特殊的符号（如[MASK]）替换掉一个或多个词，然后训练模型预测被替换掉的词。

##### GLM架构

与GPT，PaLM等模型使用Transformer的解码器方式不同，GLM-130B使用了一种双向通用语言模型（GLM）作为其Backbone。模型结构详见论文：Glm: General language model pretraining with autoregressive blank infilling (2022).

GLM是一种基于Transformer的语言模型，它以自回归空白填充为训练目标。简而言之，对于一个文本序列`x=[x1, · · · ,xn]`，从其中采样文本`span{s1，· · ·，sm}`，其中每个si表示连续令牌的跨度，并用单个掩码替换si，要求模型对它们进行自回归恢复。与GPT类模型不同的是，它在不Mask的位置使用双向注意力，因此它混合了两种Mask，以支持理解和生成：

[MASK]：句子中的短空白，长度加总到输入的某一部分
[gMASK]：随机长度的长空白，加在提供前缀上下文的句子末尾

使用以下方式实现了自回归空白填充目标。

输入 ![\bm{x}](https://www.zhihu.com/equation?tex=%5Cbm%7Bx%7D&consumer=ZHI_MENG) 被分成两部分：Part A 是损坏的文本 ![\bm{x}_{\text{corrupt}}](https://www.zhihu.com/equation?tex=%5Cbm%7Bx%7D_%7B%5Ctext%7Bcorrupt%7D%7D&consumer=ZHI_MENG)，Part B 是被遮盖的片段。Part A 的词可以相互看到，但不能看到 Part B 中的任何词。Part B 的词可以看到 Part A 和 Part B 中的前置词，但不能看到 Part B 中的后续词。为了实现自回归生成，每个片段都用特殊的符号 [START] 和 [END] 进行填充，分别用于输入和输出。这样，模型就自动地在一个统一的模型中学习了一个双向编码器（用于 Part A）和一个单向解码器（用于 Part B）。

![image-20240307213358526](..\..\img\llm-basic\glm.png)

1. 原始文本 ![\bm{x}=\left[ x_1,x_2,x_3,x_4,x_5,x_6 \right]](https://www.zhihu.com/equation?tex=%5Cbm%7Bx%7D%3D%5Cleft%5B+x_1%2Cx_2%2Cx_3%2Cx_4%2Cx_5%2Cx_6+%5Cright%5D&consumer=ZHI_MENG) 随机进行连续 mask，这里假设 mask 掉 ![\left[ x_3 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_3+%5Cright%5D&consumer=ZHI_MENG) 和 ![\left[ x_5,x_6 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_5%2Cx_6+%5Cright%5D&consumer=ZHI_MENG) ，跨度的长度服从泊松分布（ ![\lambda = 3](https://www.zhihu.com/equation?tex=%5Clambda+%3D+3&consumer=ZHI_MENG) ），与 BART 一样。
2. 将 ![\left[ x_3 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_3+%5Cright%5D&consumer=ZHI_MENG) 和 ![\left[ x_5,x_6 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_5%2Cx_6+%5Cright%5D&consumer=ZHI_MENG) 替换为 [M] 标志，并打乱 Part B 的顺序。为了捕捉跨度之间的内在联系，随机交换跨度的顺序。
3. GLM 自回归地生成 Part B。 每个片段在输入时前面加上 [S]，在输出时后面加上 [E]。 二维位置编码表示不同片段之间和片段内部的位置关系。
4. **自注意力掩码**。 **灰色区域被掩盖**。 **Part A 的词语可以自我看到（图蓝色框），但不能看到 Part B。 Part B 的词语可以看到 Part A 和 Part B 中的前面的词语（黄色和绿色框对应两个片段）**。 [M] := [MASK]，[S] := [START]，[E] := [END]。

这里解释下图中 Position1 = [1, 2, 3, 4, 5, 5, 5, 5, 3, 3]，Position2 = [0, 0, 0, 0, 0, 1, 2, 3, 1, 2] 是怎么得到的。

Position1 和 Position2 是输入的二维编码，第一个维度表示片段在原始文本中的相对位置，第二个维度表示片段内部的相对位置。

假设原始文本是 ![\bm{x}=\left[ x_1,x_2,x_3,x_4,x_5,x_6 \right]](https://www.zhihu.com/equation?tex=%5Cbm%7Bx%7D%3D%5Cleft%5B+x_1%2Cx_2%2Cx_3%2Cx_4%2Cx_5%2Cx_6+%5Cright%5D&consumer=ZHI_MENG) ，其中![\left[ x_3 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_3+%5Cright%5D&consumer=ZHI_MENG) 和 ![\left[ x_5,x_6 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_5%2Cx_6+%5Cright%5D&consumer=ZHI_MENG) 被挖去。那么，被挖去的片段在第一个维度上的位置编码就是它们在原始文本中的索引，即 ![\left[ x_3 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_3+%5Cright%5D&consumer=ZHI_MENG)来自片段 3，![\left[ x_5,x_6 \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+x_5%2Cx_6+%5Cright%5D&consumer=ZHI_MENG) 来自片段 5。在第二个维度上的位置编码就是它们在片段中的索引，即 0 和 1。因此， ![x_3](https://www.zhihu.com/equation?tex=x_3&consumer=ZHI_MENG) 的二维位置编码是[3, 0]， ![x_5](https://www.zhihu.com/equation?tex=x_5&consumer=ZHI_MENG) 的二维位置编码是[5, 0]， ![x_6](https://www.zhihu.com/equation?tex=x_6&consumer=ZHI_MENG) 的二维编码是[5, 1]。

同样，我们可以得到 ![x_1](https://www.zhihu.com/equation?tex=x_1&consumer=ZHI_MENG) 的二维位置编码是[1, 0]， ![x_2](https://www.zhihu.com/equation?tex=x_2&consumer=ZHI_MENG) 的位置编码是[2, 0]， ![x_4](https://www.zhihu.com/equation?tex=x_4&consumer=ZHI_MENG) 的位置编码是[4, 0]。

GLM 的编码方法确保了模型在重建被遮盖的跨度时不知道它们的长度。这与其他模型相比是一个重要的区别。例如，XLNet 编码了原始位置，以便它可以感知缺失的 token 的数量，而 SpanBERT 用多个 [MASK] 标记替换了跨度，并保持了长度不变。GLM 的设计适合下游任务，因为通常生成的文本的长度是事先未知的。

明白了**二维位置编码**和**自注意力掩码**，就算是明白了 GLM 的核心部分。



## 5. LLM的训练方式

### 5.1 Zero redundancy optimizer

ZeRO-DP是一种通过将**内存占用划分到多张卡或者多个节点**的支持超大规模模型训练的数据并行技术，是DeepSpeed库的核心功能之一。**传统的数据并行或者模型并行方法要么会大幅度降低计算效率，要么对内存的节省不是很友好**， 而ZeRO-DP在实现数据并行高效计算的同时，拥有模型并行的内存节省优势，很好地优化了模型状态内存

<img src="..\..\img\llm-basic\zero_3_stage.png" alt="image-20240307201230262" style="zoom:67%;" />

> Mixed precision (fp16/32) training, where parameters and activations are stored as fp16, enabling the use of the high throughput tensor core units on these GPUs. During **mixed-precision training**, both the forward and backward propagation are performed using **fp16 weights and activations**. However, to effectively compute and apply the updates at the end of the backward propagation, **the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states**.
>
>  Let’s take Adam as a concrete example. Mixed precision training of a model with Ψ parameters using Adam requires enough memory to hold an fp16 copy of the parameters and the gradients, with memory requirements of 2Ψ and 2Ψ bytes respectively. In addition, it needs to hold the optimizer states: an fp32 copy of the parameters, momentum and variance, with memory requirements of 4Ψ, 4Ψ, and 4Ψ bytes, respectively. Let’s use K to denote the memory multiplier of the optimizer states, i.e., the additional memory required to store them is KΨ bytes. Mixed-precision Adam has K = 12. In total, this results in 2Ψ + 2Ψ + KΨ = 16Ψ bytes of memory requirement. For a model such as GPT-2 with 1.5 Billion parameters, this leads to a memory requirement of at least 24 GB, which is significantly higher than the meager 3 GB of memory required to hold the fp16 parameters alone.

*ZeRO-DP* 主要有三个优化阶段，分别对应了模型状态中优化器状态、梯度，以及模型参数的切分，也就是通常所说的**ZeRO-1/2/3**

**ZeRO-1：** 优化器状态切分（ $P_{os}$ ）：切分优化器状态到各个计算卡中，在享有与普通数据并行相同通信量的情况下，可降低4倍的内存占用。

**ZeRO-2：** 添加梯度切分（$P_{os+g}$）：在 $P_{os}$ 的基础上，进一步将模型梯度切分到各个计算卡中，在享有与普通数据并行相同通信量的情况下，拥有8倍的内存降低能力。

**ZeRO-3：** 添加参数切分（ $P_{os+g+p}$ ）：在$P_{os+g}$的基础上，将模型参数也切分到各个计算卡中，内存降低能力与并行数量成线性比例，通信量大约有50%的增长。

通常模型会使用float32(fp32)精度进行训练，但是随着模型越来越大，训练的硬件成本和时间成本急剧增加。而混合精度训练通过利用float16(fp16)的优点并规避缺点来进行训练。

**优点：**

1.降低显存占用，float16比float32小一半；

2.减少网络通信开销；

3.硬件针对fp16优化，速度更快

**缺点：**

1.**下溢**。对于深度学习来说，float16最大的问题是"下溢"。模型的更新通常是 ，随着模型的训练，这个值往往会很小，可能会超出float16表示的精度。结果就是：大多数的模型权重都不再更新，模型难以收敛。

2.**舍入误差**。模型权重和梯度相差太大，通过梯度更新权重并进行舍入时，可能导致更新前和更新后的权重没有变化。

#### offload

在中间变量产生时，将中间变量移动到 CPU/NVMe 上，在需要使用中间变量时移动到 GPU 上。通过这种方式，可以减小中间变量的显存占用。Zero的Offload优化通常更适用于资源受限，但是又要训练大模型的情况。**通过时间换空间**。比如把optimizer state, parameters offload到CPU/NVMe，会有一些额外的时间开销

#### 如何选择最佳性能的ZeRO阶段和offload方式

一般而言，以下规则适用：

从速度角度来看 **Stage 0 (DDP) > Stage 1 > Stage 2 > Stage 2 + offload > Stage 3 > Stage 3 + offloads**

从GPU内存使用角度来看 **Stage 0 (DDP) < Stage 1 < Stage 2 < Stage 2 + offload < Stage 3 < Stage 3 + offloads**

因此，当想要在适合最少数量的GPU的情况下获得最快的执行速度时，可以遵循以下过程。我们从最快的方法开始，如果遇到GPU内存不足，然后转到下一个速度较慢但使用更少GPU内存的方法，依此类推。

#### 具体方法

首先，将批量大小设置为1

启用 --gradient_checkpointing 1（HF Trainer）或直接使用 model.gradient_checkpointing_enable() - 如果出现内存不足（OOM），则

首先尝试使用**ZeRO stage2**。如果出现内存不足（OOM），则

尝试使用**ZeRO stage2+ offload optimizer** - 如果出现内存不足（OOM），则

切换到**ZeRO stage3** - 如果出现内存不足（OOM），则

将 **offload_param** 启用到CPU - 如果出现内存不足（OOM），则

将 **offload_optimizer** 启用到CPU - 如果出现内存不足（OOM），则

使用混合精度进行训练而不是fp32

如果仍然出现内存不足（OOM），可以添加更多硬件或启用ZeRO-Infinity - 即将卸载 offload_param 和 offload_optimizer 切换到nvme。

#### 一些问题

为了在多个节点之间通信，DeepSpeed会要求提供一个hostfile文件，里面记录了各个节点的地址以及可用的GPU数目，然而其中重要的一点是，这些节点必须可以**相互实现无密码ssh**。出于AZML的认证保护机制，我们不能直接在AZML的集群上直接使用DeepSpeed，而需要做一些处理。

ZeRO stage 3 提供了模型参数划分，该策略会将模型的参数分配到多张显卡上。因此，当使用 `save_pretrained` 方法保存模型的checkpoint时，会出现本该完整保存的LoRA `adapter_model.bin` 被截断，只保存了一张显卡上的参数，这样便无法在之后merge微调后的模型。该问题在[此处](https://github.com/huggingface/peft/issues/460)被讨论。

该问题出现的原因可能在于：deepspeed并没有收集全部的adapter的数据，因此使用 `save_pretrained` 方法时，只保存了一张显卡（可能是rank为0的显卡）上的adapter权重，而剩余部分全是空的。

解决该问题的步骤如下：

1. 在deepspeed的config文件中，设置 `"stage3_gather_16bit_weights_on_model_save": true`，该配置会在保存时收集完整模型的全部权重（此处的完整模型是指**Base Model + LoRA**）

2. 提取完整模型的全部参数至CPU

3. 将adapter的参数从完整的 `state_dict` 中提取出来

4. 再次保存 `adapter_model.bin`，覆盖掉先前不完整的参数



### 5.2 LoRA

大型语言模型很大，并且由于 GPU 内存限制，在训练期间更新所有模型权重的成本可能会很高。例如，假设我们有一个 LLM，其 7B 个参数以权重矩阵*W表示*。（实际上，模型参数当然分布在多层的不同矩阵中，但为了简单起见，我们在这里指的是单个权重矩阵）。在反向传播过程中，我们学习一个*ΔW*矩阵，其中包含**我们想要更新多少原始权重以在训练期间最小化损失函数的信息**。
那么权重更新如下：
*W*更新= *W* + *ΔW*
如果权重矩阵*W*包含7B个参数，则权重更新矩阵*ΔW*也包含7B个参数，**并且计算矩阵*ΔW*可能是计算量和内存密集型的**。
[Hu](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.09685)*[等人](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.09685)*提出的LoRA方法。替换为**将权重变化*ΔW分解*为较低等级的表示**。准确地说，它不需要显式计算*ΔW* 。*相反，LoRA 在训练期间直接学习ΔW*的分解表示，这就是节省的来源，如下图所示。

<img src="..\..\img\llm-basic\lora.png" alt="img" style="zoom:80%;" />

如上所示，*ΔW的分解*意味着我们用两个较小的 LoRA 矩阵*A*和*B来表示大矩阵ΔW* 。如果*A*的行数与*ΔW相同*，*B的列数与ΔW*相同，我们可以将分解写为*ΔW = AB* 。( *AB是矩阵A*和*B*之间的矩阵乘法结果。)

当然，*A*和*B*无法捕获*ΔW*可以捕获的所有信息，但这是设计使然。当使用 LoRA 时，我们假设模型需要*W*是一个满秩的大矩阵，以捕获预训练数据集中的所有知识。然而，当我们微调LLM时，我们不需要更新所有权重并以比*ΔW更少的权重捕获适应的核心信息*；*因此，我们通过AB*进行低等级更新。





## Mistral

Mistral 7B利用了分组查询注意力（GQA）和滑动窗口注意力（SWA）。

GQA显著提高了推断速度，同时在解码过程中减少了内存需求，允许使用更大的批处理大小，因此**提高了吞吐量**，这对实时应用来说是一个至关重要的因素。

此外，**SWA设计用于更有效地处理更长的序列**，以降低计算成本，从而缓解了语言模型的一个普遍限制。这些注意机制共同为Mistral 7B的增强性能和效率做出了贡献。

**Sliding Window Attention**。在传统的注意力机制中，vanilla attention中的操作次数与序列长度的平方成正比，而内存随着token数量的线性增加。在推理时，由于缓存可用性减少，这导致更高的延迟和较小的吞吐量。为了缓解这个问题，我们使用sliding window attention：每个token最多可以关注前一层的W个token（这里，W = 3）。请注意，滑动窗口外的token仍然会影响下一个单词的预测。在每个attention层，信息可以向前移动W个token。因此，在k个attention层之后，信息可以向前移动最多k × W个token。

<img src="..\..\img\llm-basic\mistral_swa.png" alt="image-20240307225349840" style="zoom:50%;" />
