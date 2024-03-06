## Freq Q&A

### 1. Multi-head self-attention

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
    # Output:  [batch_size, seq_len, d_model]
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
    	nbatches = query.size(0)

    	# 1) Do all the linear projections in batch from d_model => h*d_k
    	query, key, value = [
			lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
      		for lin, x in zip(self.linears, (query, key, value))
    	]
        # query, key, value: [batch_size, num_heads, seq_len, d_model]

    	# 2) Apply attention on all the projected vectors in batch.
        # x: [batch_size, num_heads, seq_len, d_model]
    	x, self.attn = attention(
			query, key, value, mask=mask, dropout=self.dropout
    	)
        

		# 3) "Concat" using a view and apply a final linear.
    	x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
    	del key
    	del value
    	return self.linears[-1](x)
```



### 2. Post-Norm和Pre-Norm

Bert的Post-Norm，是在Add操作后进行Norm操作，因此叫做Post-Norm

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



### 3. Layer Norm 与 Batch Norm

BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。**BN的具体做法就是对每一小批数据，在批这个方向上做归一化**。

* 对每一个batch进行操作，使得对于这一个batch中所有的输入数据，它们的每一个特征都是均值为0，方差为1的分布 

* 单纯把所有的输入限制为(0,1)分布也是不合理的，这样会降低数据的表达能力（第L层辛苦学到的东西，这里都暴力变成(0,1)分布了）。因此需要再加一个线性变换操作，让数据恢复其表达能力。这个线性变化中的两个参数 $\gamma$, $\beta$ 是需要模型去学习的。

LN整体做法类似于BN，不同的是LN不是在特征间进行标准化操作（横向操作），而是在整条数据间进行标准化操作**（纵向操作）**。它也是归一化数据的一种方式，不过**LN 是在每一个样本上计算均值和方差**，而不是BN那种在批方向计算均值和方差

<img src="https://raw.githubusercontent.com/oraccc/NLP-Basic/master/img/transformer/normalization.png" width="650" />
