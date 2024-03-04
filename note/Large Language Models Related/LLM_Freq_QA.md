## Freq Q&A

### Multi-head self-attention

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

