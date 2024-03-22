## LLaMA模型结构

LLaMA V1 和 V2 模型结构基本相同，主要由 Attention 和 MLP 堆叠而成

<img src="..\..\img\llm-basic\llama.png" alt="Image" style="zoom:50%;" />

### LLaMA V1主要特点

- 前置的 RMSNorm；
- 在Q、K上使用旋转式位置编码 RoPE；
- 使用 Causal Mask 保证每个位置只能看到前面的 Tokens；
- 可以将更早的 K、V 拼接到当前 K、V 前面，可以用 Q 查找更早的信息（为了清晰没在图中画出来）；
- MLP表达式：$down(up(x) \times SiLU(gate(x)))$，其中 down、up、gate 都是线性层 Linear。

#### RMSNorm

BERT、GPT 等模型中广泛使用的是 LayerNorm
$$
y=\frac{x-Mean(x)}{\sqrt{Var(x)+\epsilon}}*W+B
$$


RMSNorm（Root Mean Square Layer Normalization）发现 LayerNorm 的中心偏移没什么用（减去均值等操作）。将其去掉之后，效果几乎不变，但是速度提升了40%。RMSNorm 最终公式变为：
$$
y=\frac{x}{\sqrt{Mean(x^2)+\epsilon}}*W
$$
注意除了没有减均值、加偏置以外，分母上求的 RMS 而不是方差。

**LLaMA 在 Attention Layer 和 MLP 的输入上使用了 RMSNorm，相比在输出上使用，训练会更加稳定。**

#### SwiGLU

LLaMA 没有使用 ReLU，而是使用了 SwiGLU，有时也被称为 SiLU。公式为：
$$
Sigmoid(x)*x
$$
SwiGLU 的效果类似平滑版的 ReLU，如下图所示：

<img src="..\..\img\llm-basic\silu.png" alt="Image" style="zoom:67%;" />

#### RoPE

LLaMA 使用的位置编码是 RoPE（Rotary Position Embedding）

使用了这么复杂的位置编码，有什么好处呢？

从上面**RoPE 形式上是绝对位置编码，即依赖其绝对位置m；但当我们计算 Attention 时，RoPE 却可以变成相对位置编码。**绝对位置编码的优点是计算速度快等，缺点是拓展长度比较麻烦，且绝对位置并没有什么实际意义。

而**相对位置编码对学习 token 之间的关系很有意义**，比如距离的很远的两个 token 之间的关联大概率很小，使用相对位置编码往往能够获得更好的效果。此外**相对位置编码拓展长度也更容易**，因为不论 context size 多长，只需关注最长距离以内的输入即可。

相对位置编码的缺点是没有绝对位置编码计算速度快。

q 和 k 的 attention 依赖相对距离 m-n。因此 RoPE 为 q、k 注入的绝对位置编码，计算得到的 attention，却变成了相对位置编码

### LLaMA V2 相对 V1 的更新

1. 预训练语料从 1 Trillion tokens -> 2 Trillion tokens；
2. context window 长度从 2048 -> 4096；
3. 收集了 100k 人类标注数据进行 SFT；
4. 收集了 1M 人类偏好数据进行RLHF；
5. 在 reasoning, coding, proficiency, and knowledge tests 上表现超越 MPT 和 Falcon；
6. 和 Falcon 模型一样，**使用了 Group Query Attention，节省 cache。**

#### Group Query Attention

当前有如下 3 种主流的 Attention 计算方式：

<img src="..\..\img\llm-basic\llm_3_attention.png" alt="Image" style="zoom:50%;" />

自回归模型生成回答时，**需要前面生成的 KV 缓存起来，来加速计算**。

Multi-Head Attention（MHA）就是多个头各自拥有自己的 Q,K,V 来算各自的 Self-Attention，需要的缓存量很大。

Multi-Query Attention （MQA）指出多个头之间可以共享 KV 对，即 Q 依然保持多头，但是 KV 只有一个。

Group Query Attention （GQA）没有像 MQA 一样极端，将 Query 分组，组内共享 KV，效果接近 MHA，速度上与 MQA 可比较。