

> LLM 基于的 Transfomer 模型不再使用基于循环的方式建模文本输入，**序列中不再有任何信息能够提示模型单词之间的相对位置关系。**
> Transformer 模型在处理序列数据时，其**自注意力机制使得模型能够全局地捕捉不同元素之间的依赖关系**，但这样做的代价是丧失了序列中的元素顺序信息。由于自注意力机制并不考虑元素在序列中的位置，所以在输入序列的任何置换下都是不变的，这就意味着**模型无法区分序列中元素的相对位置**。在许多自然语言处理任务中，**词语之间的顺序是至关重要的**，所以需要一种方法来让模型捕获这一信息。
> 因此在送入编码器端建模其上下文语义之前，一个非常重要的操作是**在词嵌入中加入位置编码（Positional Encoding）**这一特征。
> 具体来说，**序列中每一个单词所在的位置都对应一个向量**。这一**向量会与单词表示对应相加**并送入到后续模块中做进一步处理。在训练的过程当中，**模型会自动地学习到如何利用这部分位置信息。**

常见的位置编码主要有绝对位置编码（sinusoidal），旋转位置编码（RoPE），以及相对位置编码ALiBi

## 绝对位置编码sinusoidal

绝对位置编码是直接将序列中每个位置的信息编码进模型的，从而使模型能够了解每个元素在序列中的具体位置。**原始Transformer提出时采用了sinusoidal位置编码，通过使用不同频率的正弦和余弦的函数，使得模型捕获位置之间的复杂关系，且这些编码与序列中每个位置的绝对值有关。**

sinusoidal位置编码公式如下：
$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$
其中，$pos$表示位置，$d_{model}$代表embedding的总维度，$2i, 2i+1$代表的是embedding不同位置的索引，即位置编码向量对应的维度位置。

> 举一个最经典的ViT（Vision Transformer）的例子：假设使用的是ViT-Large这个模型，输入的图像大小是224x224，Patch大小是16，则经过Patch Embedding后会得到（196, 1024）的Tensor，此时构建位置编码，$d_{model}$就等于1024，$pos$的取值范围就是0-195（range(196)），$2i$ 和 $2i+1$ 的取值范围就是0-1023（range(1024)）。

原始 Transformer 的位置编码虽然是基于绝对位置的，但其数学结构使其能够捕获一些相对位置信息。使用正弦和余弦函数的组合为每个位置创建编码，波长呈几何级数排列，意味着每个位置的编码都是独特的。同时，正弦和余弦函数的周期性特性确保了不同位置之间的编码关系是连续且平滑的。

比如：

- 对于相邻位置，位置编码的差异较小，与两者之间的距离成正比。
- 对于相隔较远的位置，位置编码的差异较大，与两者之间的距离也成正比。

这种连续和平滑的关系允许模型学习位置之间的**相对关系**，而不仅仅是各自的绝对位置。考虑两个位置 $i$ 和 $j$，由于正弦和余弦函数的性质，位置编码的差值 $PE(i)-PE(j)$ 将与和之间的差值有关。这意味着通过比较不同位置编码之间的差值，模型可以推断出它们之间的相对位置。

总结来说，通过上面这种方式计算位置编码有这样几个好处：

- **首先，正余弦函数的范围是在 [-1,1]，导出的位置编码与原词嵌入相加，不会使得结果偏离过远而破坏原有单词的语义信息。**
- **其次，依据三角函数的基本性质，可以得知第 pos + k 个位置的编码是第 pos 个位置的编码的线性组合，这就意味着位置编码中蕴含着单词之间的距离信息。**



### PyTorch代码实现

```python
import math
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model:int, max_seq_len:int = 100):
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

<img src="..\..\img\llm-basic\sinusoidal.png" alt="图片" style="zoom:50%;" />



## 旋转位置编码RoPE

sinusoidal位置编码对相对位置关系的表示还是比较间接的，那有没有办法**更直接的表示相对位置关系**呢？那肯定是有的，而且有许多不同的方法，旋转位置编码（Rotary Position Embedding，RoPE）是一种用绝对位置编码来表征相对位置编码的方法，并被用在了很多大语言模型的设计中，很多成功的LLM，例如LLAMA系列、GLM、百川、通义千问等，都使用了RoPE。

RoPE 借助了复数的思想，出发点是**通过绝对位置编码的方式实现相对位置编码。**

RoPE的设计思路可以这么来理解：我们通常会通过向量$q$和$k$的内积来计算注意力系数，如果能够对 $q$、$k$ 向量注入了位置信息，然后用更新的 $q$、$k$ 向量做内积就会引入位置信息了。

假设 $f(q,m)$ 表示给在位置 $m$ 的向量 $q$ 添加位置信息的操作，如果**叠加了位置信息后的 $q$（位置 $m$ ）和 $k$（位置 $n$ ）向量的内积可以表示为它们之间距离的差m-n的一个函数，那不就能够表示它们的相对位置关系了**。也就是我们希望找到下面这个等式的一组解：
$$
<f(q,m), f(k,n)>=g(q,k,m-n)
$$
RoPE这一研究就是为上面这个等式找到了一组解答，也就是
$$
f(q,m)=qe^{im\theta} \\
f(k,n)=ke^{in\theta}
$$
<img src="..\..\img\llm-basic\rope_math.png" alt="image-20240305000448536" style="zoom:67%;" />

有了这一形式后，具体实现有两种方式：

- 转到复数域，对两个向量进行旋转，再转回实数域
- 由于上述矩阵 Rn 具有稀疏性，因此可以使用逐位相乘 ⊗ 操作进一步加快计算速度，直接在实数域通过向量和正余弦函数的乘法进行运算
- <img src="..\..\img\llm-basic\rope_math2.png" alt="Image" style="zoom: 50%;" />

```python
import torch
import math

def rotary_position_embedding(q, k):
    """
    Rotary Position Embedding (RoPE) for queries and keys.
    
    Args:
        q: tensor for queries of shape (batch_size, num_heads, seq_len, dim)
        k: tensor for keys of shape (batch_size, num_heads, seq_len, dim)
        
    Returns:
        Rotated queries and keys
    """
    batch_size, num_heads, seq_len, dim = q.size()
    
    # Begin of sinusoidal_position_embedding content
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)).to(q.device)
    
    pos_emb = position * div_term
    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    pos_emb = pos_emb.unsqueeze(0).unsqueeze(1)
    pos_emb = pos_emb.expand(batch_size, num_heads, -1, -1)
    # End of sinusoidal_position_embedding content

    # Extract and duplicate cosine and sine embeddings
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # Create alternate versions of q and k
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    k_alternate = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.size())

    # Rotate queries and keys
    q_rotated = q * cos_emb + q_alternate * sin_emb
    k_rotated = k * cos_emb + k_alternate * sin_emb

    return q_rotated, k_rotated
```

