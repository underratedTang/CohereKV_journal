# \method：动机、方法流程与理论保证

## 1. 动机

Merging-based KV-cache compression 的目标是在固定缓存预算下保留待删除 tokens 的信息。现有工作已经证明 token merging 可以优于直接 eviction，但其 grouping 机制主要优化局部匹配质量或计算效率，而不是认证一个多成员 group 中全部 keys 的两两一致性。

**CentroidKV** 将序列分块，在每个 chunk 内交替划分集合 \(A,B\)，再令每个 \(A\) 中 token 连接到最相似的 \(B\) 中 token，仅保留排名靠前的 matching edges，并将相连 tokens 合并为 centroid。该设计将距离计算从全序列 all-pairs 降为 chunked bipartite matching，但显式判定对象是 source–destination edge。若多个 source tokens 指向同一 destination，算法没有进一步认证这些 source tokens 彼此兼容；多轮 centroid clustering 也未维护原始成员级的完整组直径证书。[^centroidkv]

**D2O** 先将 tokens 划分为 conserved 和 evicted sets，再为每个 evicted key 通过 cosine similarity 选择最相似的 conserved key。evicted–conserved similarity 超过动态 EMA threshold 时，evicted KV 被加权合并到该 target。该 many-to-one 规则判断每个 evicted token 与 target 的关系，但没有判断分配到同一 target 的不同 evicted tokens 之间的关系，也没有跨轮传播能够约束历史原始成员的 group certificate。[^d2o]

**SelKV** 将每个 evicted token 路由到 positional bucket 内的 kept target，再根据该 evicted token 与 target 的 value-vector cosine similarity 设置 soft merge gate。该机制能够抑制单个不合适 token 对 target 的污染，但 gate 仍是 token–target 局部判定；多个 evicted tokens 合并到同一 target 时，它不认证这些 tokens 的 keys 两两兼容。[^selkv]

**SemantiCache** 在 delimiter-defined semantic chunk 内执行 Greedy Seed-Based Clustering。每个未分配 token 成为 seed，所有与该 seed 的 key cosine similarity 超过阈值的后续 tokens 被加入同一 cluster。该判定只约束 seed–member 关系；两个成员可以分别接近 seed，却彼此不接近。由于算法按序选择 seed 并执行单次顺序扫描，cluster assignment 还依赖 seed 和 token 的处理顺序。[^semanticache]

上述方法并非无效；它们分别服务于低开销 matching、动态补偿、per-token selective merging 和 semantic chunk preservation。本文关注的是它们未解决的另一项要求：

\[
\boxed{
\text{给定一个 retained anchor 和多个 evicted entries，
认证最终 group 所代表的全部原始 keys 两两满足统一距离约束。}
}
\]

这一要求不能由独立的 anchor–candidate 或 seed–member 判定推出。若显式构造 candidate–candidate compatibility，寻找包含最多 candidates 的合法 group 等价于 maximum clique 或 maximum independent set，在一般图上是 NP-hard；逐个加入 candidate 的 complete-link greedy 又产生串行依赖。缓存项还可能已经代表多个历史 tokens，算法不能展开原始成员重新计算 all-pairs 距离，必须维护可跨轮传播的紧凑证书。不同模型、层和 KV head 的 key-distance 尺度也不同，绝对阈值不能直接迁移。

\method 将该问题分解为四步：以在线归一化欧氏距离定义统一质量预算；用 \(R,D\) 证书认证 entry pairs；将同一 retained anchor 的 evicted candidates 转化为局部冲突图；使用 degree-priority parallel maximal independent set 生成 membership mask。retained anchor 固定保留，selected evicted entries 在冲突图中两两无边，因此最终 group 获得完整的原始 key-pair diameter guarantee，而无需展开历史成员或求解 NP-hard 的 maximum group。

[^centroidkv]: Jie Hu et al., *CentroidKV: Efficient Long-Context LLM Inference via KV Cache Clustering*, arXiv:2506.11418, 2025/2026.
[^d2o]: Zhongwei Wan et al., *D2O: Dynamic Discriminative Operations for Efficient Long-Context Inference of Large Language Models*, arXiv:2406.13035; ICLR 2025.
[^selkv]: Soumia Bouyahiaoui et al., *SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation*, arXiv:2607.16213, 2026.
[^semanticache]: Shunlong Wu et al., *SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging*, arXiv:2603.14303, 2026.

---

## 2. 方法目标与基本符号

\method 的输入是外部 selector 给出的 retained 集合与 evicted 集合。算法首先用 entry-pair certificate 将每个 evicted entry 路由到最近的兼容 retained anchor；随后在每个 anchor 的 evicted candidate pool 内构造冲突图，并用 degree-priority parallel MIS 选出 maximal independent set。retained anchor 始终固定保留，MIS 只生成 evicted membership mask。由于 accepted candidates 分别与 anchor 兼容，且它们在冲突图中两两无边，最终 group 所代表的全部原始 keys 两两满足欧氏直径阈值。选中 entries 再通过 count-weighted arithmetic mean 合并，\(C,R,D\) 负责保存成员数及跨轮证书。


在压缩轮次 \(r\)，\method 的必要缓存状态表示为

\[
\left(
K^{(r)},V^{(r)},C^{(r)},R^{(r)},D^{(r)}
\right),
\]

其中

\[
K^{(r)},V^{(r)}
\in
\mathbb{R}^{B_{\mathrm{sz}}\times H_{kv}\times L^{(r)}\times d},
\]

\[
C^{(r)}
\in
\mathbb{N}_{+}^{B_{\mathrm{sz}}\times H_{kv}\times L^{(r)}},
\]

\[
R^{(r)},D^{(r)}
\in
\mathbb{R}_{+}^{B_{\mathrm{sz}}\times H_{kv}\times L^{(r)}}.
\]

令 \(\mathcal S_i^{(r)}\) 表示缓存项 \(i\) 所代表的原始 token 集合；该集合仅用于理论定义，实际执行时不存储。初始时，

\[
\mathcal S_i^{(0)}=\{i\}.
\]

任意轮次中，当前不同缓存项所代表的原始 token 集合两两不交。缓存项满足

\[
C_i^{(r)}
=
\left|\mathcal S_i^{(r)}\right|,
\]

\[
K_i^{(r)}
=
\frac{1}{C_i^{(r)}}
\sum_{x\in\mathcal S_i^{(r)}}k_x,
\qquad
V_i^{(r)}
=
\frac{1}{C_i^{(r)}}
\sum_{x\in\mathcal S_i^{(r)}}v_x.
\]

等价地，

\[
C_i^{(r)}K_i^{(r)}
=
\sum_{x\in\mathcal S_i^{(r)}}k_x,
\qquad
C_i^{(r)}V_i^{(r)}
=
\sum_{x\in\mathcal S_i^{(r)}}v_x.
\]

\(R_i^{(r)}\) 是原始成员到当前 compressed centroid 的半径证书：

\[
R_i^{(r)}
\geq
\max_{x\in\mathcal S_i^{(r)}}
\left\|k_x-K_i^{(r)}\right\|_2.
\]

\(D_i^{(r)}\) 是该缓存项内部原始 keys 的直径证书：

\[
D_i^{(r)}
\geq
\max_{x,y\in\mathcal S_i^{(r)}}
\left\|k_x-k_y\right\|_2.
\]

利用 count 可在不增加状态的情况下收紧半径证书。定义

\[
\boxed{
\widehat R_i^{(r)}
=
\min\left(
R_i^{(r)},
\frac{C_i^{(r)}-1}{C_i^{(r)}}D_i^{(r)}
\right).
}
\]

附录 A.1 证明 \(\widehat R_i^{(r)}\) 仍是原始成员到当前 centroid 的有效距离上界。新生成 token 初始化为

\[
C_i=1,
\qquad
R_i=0,
\qquad
D_i=0.
\]

全文中的 \(k_x\) 和 \(K_i^{(r)}\) 均指实际参与 attention dot product 的 key 表示。对于采用 RoPE 的模型，它们是已经应用 RoPE 后并存入 KV cache 的 keys；合并也直接在该缓存 key 空间中执行。

\((K,V,C,R,D)\) 是 certificate-based grouping core 的必要状态。selector-specific state 不属于核心状态；本文默认 selector 及其位置元数据 \(p\) 和 recent-score history \(P\) 见附录 B。

欧氏距离归一化另外使用请求内、每层每个 KV head 的在线统计状态。首轮压缩前暂时维护原始 key 的样本数 \(n\)、均值向量 \(\mu\) 和中心化平方和 \(M_2\)；首轮压缩时将其转换为固定参考尺度 \(S^{\mathrm{ref}}\)，随后可释放 \(\mu,M_2\)，仅保留一个标量 \(S^{\mathrm{ref}}\)。该状态不是 per-entry certificate state，也不需要离线校准数据。

\method 的目标是在不显式访问 \(\mathcal S_i^{(r)}\) 的情况下，利用每个缓存项的 centroid、有效半径证书和直径证书，构造任意原始成员两两欧氏距离均受控的 groups。

---

## 3. 推理阶段在线归一化欧氏直径

对于一个由原始 keys 构成的组 \(G\)，定义其 key-space Euclidean diameter：

\[
\operatorname{diam}(G)
=
\max_{x,y\in G}
\left\|k_x-k_y\right\|_2.
\]

欧氏距离保留 key direction 和 key norm 的共同差异，并可直接进入 attention-logit deviation bound，但其绝对数值会随模型、层和 KV head 的 key scale 改变。因此，\method 不把欧氏距离映射成人为的相似度，而是在当前推理请求内自动估计一个无量纲化参考尺度。

以下对每个 Transformer layer、batch \(b\) 和 KV head \(h\) 独立执行；为简化符号，省略 layer index。设首轮压缩发生前已经产生的原始、未合并 cached keys 为

\[
k_1,\ldots,k_n.
\]

定义它们的均值和中心化平方和：

\[
\mu_{b,h}
=
\frac{1}{n}
\sum_{a=1}^{n}k_a,
\]

\[
M_{2,b,h}
=
\sum_{a=1}^{n}
\left\|k_a-\mu_{b,h}\right\|_2^2.
\]

当前请求的参考尺度定义为原始 key pairs 的 RMS Euclidean distance：

\[
\boxed{
S_{b,h}^{\mathrm{ref}}
=
\sqrt{
\frac{1}{\binom n2}
\sum_{1\leq a<c\leq n}
\left\|k_a-k_c\right\|_2^2
+
\epsilon_s^2
}.
}
\]

利用恒等式

\[
\sum_{a<c}
\left\|k_a-k_c\right\|_2^2
=
n
\sum_a
\left\|k_a-\mu_{b,h}\right\|_2^2,
\]

可等价写成

\[
\boxed{
S_{b,h}^{\mathrm{ref}}
=
\sqrt{
\frac{2M_{2,b,h}}{n-1}
+
\epsilon_s^2
},
\qquad n\geq2.
}
\]

其中 \(\epsilon_s>0\) 仅用于避免退化输入导致零参考尺度。若首轮压缩触发时 \(n<2\)，本轮不执行 grouping；实际 KV-cache 压缩通常在 \(n\) 远大于 \(2\) 时触发。

该统计量无需显式计算 all-pairs distance。首轮压缩前，每产生一个新的原始 key \(k\)，使用 Welford update：

\[
n'
=
n+1,
\]

\[
\delta
=
k-\mu,
\]

\[
\mu'
=
\mu+\frac{\delta}{n'},
\]

\[
M_2'
=
M_2+\delta^\top(k-\mu').
\]

在当前请求第一次触发压缩时，根据 \(n,\mu,M_2\) 计算 \(S_{b,h}^{\mathrm{ref}}\)，并在该请求剩余的所有压缩轮次中冻结。后续 merged centroids 和新增 decoding tokens 不再更新该参考尺度，从而避免 centroid averaging 或压缩历史使阈值跨轮次漂移。

定义 normalized group diameter：

\[
\overline{\operatorname{diam}}(G)
=
\frac{
\operatorname{diam}(G)
}{
S_{b,h}^{\mathrm{ref}}
}.
\]

certificate-based grouping 模块的唯一主要质量超参数为

\[
\boxed{
\lambda>0,
}
\]

称为 normalized Euclidean diameter threshold。当前请求中允许的最大完整组欧氏直径为

\[
\boxed{
\Delta_{b,h}^{\mathrm{ref}}
=
\lambda S_{b,h}^{\mathrm{ref}}.
}
\]

对给定阈值 \(\Delta_{b,h}^{\mathrm{ref}}\)，若一个 group 所代表的全部原始 keys 满足

\[
\max_{x,y\in G}
\left\|k_x-k_y\right\|_2
\leq
\Delta_{b,h}^{\mathrm{ref}},
\]

则称该 group 满足 \(\Delta_{b,h}^{\mathrm{ref}}\)-complete consistency。等价地，

\[
\overline{\operatorname{diam}}(G)
\leq
\lambda.
\]

\(\lambda\) 的含义是：允许的最大 certified group diameter 相对于当前请求、对应 layer 和 KV head 中典型原始 key-pair RMS distance 的倍数。因此：

- \(\lambda\) 越小，group 越紧，合并越保守；
- \(\lambda\) 越大，允许的组直径越大，合并越激进；
- \(\lambda=0\) 的极限只允许完全重合的 keys；
- 数学上不需要将 \(\lambda\) 限制在 \((0,1)\)，也不将其称为 similarity。

若一个模型的全部 keys 被统一缩放为 \(k'_x=ck_x\)，则 \(\operatorname{diam}(G)\) 和 \(S^{\mathrm{ref}}\) 同时缩放 \(c\) 倍，因此 normalized diameter 不变。该定义消除整体 key-scale 差异，同时仍保留 key norm 对 attention dot product 的影响。

本文不预设跨模型通用的固定 \(\lambda\) 默认值。实验应报告 \(\lambda\)、实际 normalized group diameter、accepted ratio、group size 和下游性能之间的关系，而不是将 \(\lambda\) 解释为人工构造的相似度分数。

---

## 4. \method 完整流程

### Step 0：冻结请求内距离尺度

首轮压缩前，以 Welford statistics 计算

\[
S_{b,h}^{\mathrm{ref}}
=
\sqrt{
\frac{2M_{2,b,h}}{n_{b,h}-1}
+
\epsilon_s^2
}.
\]

给定唯一质量参数 \(\lambda>0\)，固定

\[
\boxed{
\Delta_{b,h}^{\mathrm{ref}}
=
\lambda S_{b,h}^{\mathrm{ref}}.
}
\]

后续压缩轮次不再更新 \(S_{b,h}^{\mathrm{ref}}\)。

### Step 1：获取 retained/evicted partition

外部 selector \(\Phi\) 输出

\[
\left(
\mathcal C_{\mathrm{keep}}^{(r)},
\mathcal C_{\mathrm{evict}}^{(r)}
\right)
=
\Phi\left(
\mathcal C^{(r)},
N_{\mathrm{keep}}^{(r)}
\right),
\]

满足

\[
\mathcal C_{\mathrm{keep}}^{(r)}
\cap
\mathcal C_{\mathrm{evict}}^{(r)}
=
\varnothing,
\qquad
\mathcal C_{\mathrm{keep}}^{(r)}
\cup
\mathcal C_{\mathrm{evict}}^{(r)}
=
\mathcal C^{(r)}.
\]

默认 selector 见附录 B。

### Step 2：证书兼容路由

对任意 entries \(u,v\)，定义

\[
d_{uv}^{(r)}
=
\left\|K_u^{(r)}-K_v^{(r)}\right\|_2,
\]

\[
\boxed{
\Gamma_{uv}^{(r)}
=
\max\left\{
D_u^{(r)},
D_v^{(r)},
\widehat R_u^{(r)}
+
d_{uv}^{(r)}
+
\widehat R_v^{(r)}
\right\}.
}
\]

evicted entry \(i\) 的兼容 retained anchors 为

\[
\mathcal F_i^{(r)}
=
\left\{
j\in\mathcal C_{\mathrm{keep}}^{(r)}
:
\Gamma_{ij}^{(r)}
\leq
\Delta_{b,h}^{\mathrm{ref}}
\right\}.
\]

若 \(\mathcal F_i^{(r)}\neq\varnothing\)，定义

\[
j_i^\star
=
\min\left(
\operatorname*{arg\,min}_{j\in\mathcal F_i^{(r)}}
d_{ij}^{(r)}
\right).
\]

候选池为

\[
\boxed{
\mathcal P_j^{(r)}
=
\left\{
i\in\mathcal C_{\mathrm{evict}}^{(r)}
:
j_i^\star=j
\right\}.
}
\]

无兼容 anchor 的 evicted entry 不进入候选池。

### Step 3：冲突图与 degree-priority parallel MIS

对每个 retained anchor \(j\)，构造 evicted-only conflict graph

\[
\mathcal H_j^{(r)}
=
\left(
\mathcal P_j^{(r)},
\mathcal E_j^{(r)}
\right),
\]

\[
\boxed{
(i,\ell)\in\mathcal E_j^{(r)}
\iff
\Gamma_{i\ell}^{(r)}
>
\Delta_{b,h}^{\mathrm{ref}}.
}
\]

retained anchor \(j\) 不进入图。对 candidate \(i\)，定义固定字典序 priority

\[
\boxed{
\pi_i
=
\left(
-\deg_{\mathcal H_j}(i),
-\Gamma_{ij}^{(r)},
-\operatorname{hash}(i)
\right).
}
\]

每个并行轮次同时选取 priority 高于全部 active conflict neighbors 的 vertices，加入结果并删除其 conflict neighbors，直到无 active vertex。输出

\[
\boxed{
\mathcal Q_j^{(r)}
=
\operatorname{ParallelMIS}
\left(
\mathcal H_j^{(r)}
\right).
}
\]

membership mask 为

\[
\boxed{
M_{ij}^{(r)}
=
\mathbb I[j=j_i^\star]
\mathbb I[i\in\mathcal Q_j^{(r)}].
}
\]

### Step 4：更新成员集合

\[
G_j^{(r)}
=
\{j\}
\cup
\mathcal Q_j^{(r)},
\]

\[
\boxed{
\mathcal S_j^{(r+1)}
=
\mathcal S_j^{(r)}
\uplus
\biguplus_{i\in\mathcal Q_j^{(r)}}
\mathcal S_i^{(r)}.
}
\]

未被接受的 evicted entries 从缓存中删除。

### Step 5：Count-weighted KV reduction

\[
\boxed{
C_j^{(r+1)}
=
C_j^{(r)}
+
\sum_i
M_{ij}^{(r)}C_i^{(r)}.
}
\]

\[
\boxed{
K_j^{(r+1)}
=
\frac{
C_j^{(r)}K_j^{(r)}
+
\sum_iM_{ij}^{(r)}C_i^{(r)}K_i^{(r)}
}{
C_j^{(r+1)}
}.
}
\]

\[
\boxed{
V_j^{(r+1)}
=
\frac{
C_j^{(r)}V_j^{(r)}
+
\sum_iM_{ij}^{(r)}C_i^{(r)}V_i^{(r)}
}{
C_j^{(r+1)}
}.
}
\]

selector-specific state 使用同一 mask 独立更新，不参与 \(K,V\) 权重。

### Step 6：更新 \(R,D\) 证书

令

\[
\overline G_j^{(r)}
=
\{j\}
\cup
\mathcal Q_j^{(r)},
\]

\[
B_{uv}^{(r)}
=
\widehat R_u^{(r)}
+
d_{uv}^{(r)}
+
\widehat R_v^{(r)}.
\]

\[
\boxed{
D_j^{(r+1)}
=
\max\left\{
\max_{u\in\overline G_j^{(r)}}D_u^{(r)},
\;
\max_{\substack{u,v\in\overline G_j^{(r)}\\u<v}}
B_{uv}^{(r)}
\right\}.
}
\]

\[
\widetilde R_j^{(r+1)}
=
\max_{u\in\overline G_j^{(r)}}
\left[
\widehat R_u^{(r)}
+
\left\|K_u^{(r)}-K_j^{(r+1)}\right\|_2
\right],
\]

\[
\boxed{
R_j^{(r+1)}
=
\min\left(
\widetilde R_j^{(r+1)},
D_j^{(r+1)}
\right).
}
\]

### Step 7：Count-calibrated decoding

\[
\boxed{
\widetilde L_t
=
\frac{q_t{K^{(r)}}^\top}{\sqrt d}
+
\log C^{(r)}.
}
\]

\(\log C\) 在 logits 进入 softmax 前按 entry 相加。

---

## 5. 并行实现、复杂度与数值细节

\method 的一次压缩由以下阶段组成：

\[
\text{importance split}
\rightarrow
\text{anchor-distance GEMM}
\rightarrow
\text{nearest compatible routing}
\rightarrow
\text{local conflict-graph construction}
\rightarrow
\text{parallel MIS}
\rightarrow
\text{segmented reductions}
\rightarrow
\text{on-the-fly }\log C\text{ calibration}.
\]

设

\[
N_e
=
\left|
\mathcal C_{\mathrm{evict}}^{(r)}
\right|,
\qquad
N_r
=
\left|
\mathcal C_{\mathrm{keep}}^{(r)}
\right|,
\]

并令

\[
m_j
=
\left|
\mathcal P_j^{(r)}
\right|,
\qquad
\sum_jm_j\leq N_e.
\]

主要计算工作量为：

- evicted–retained anchor-distance GEMM：
  \[
  O(N_eN_rd);
  \]
- anchor compatibility 和 nearest compatible routing：
  \[
  O(N_eN_r);
  \]
- 每个局部候选池的 candidate–candidate distance：
  \[
  O\left(
  d\sum_jm_j^2
  \right);
  \]
- parallel MIS：每一轮的工作量与当前 active conflict edges 或其 bitset 表示成正比；不同 anchors 和同一轮的不同 vertices 均可并行；
- masked count-weighted reductions 和证书归约：与 selected entries 及 selected pair certificates 的数量成线性关系。

相对于 CentroidKV、D2O、SelKV 和 SemantiCache 的局部关系判定，局部图方法的最坏 candidate–candidate 工作量为 \(O(N_e^2d)\)，因此其总工作量高于只计算 evicted–retained 局部匹配关系的方法。候选池由 retained routing 划分，局部距离由 batched GEMM 并行计算，MIS 在每轮同时接受多个无冲突 candidates。该计算用于获得现有 source–target 或 seed–member grouping 没有提供的完整组两两认证。

寻找 maximum group 需要解 maximum independent set，不作为在线算法目标。Degree-priority parallel MIS 返回 maximal group：最终每个未选择 candidate 要么与某个 selected candidate 冲突，要么没有兼容 retained anchor；因此不能在不破坏当前完整组保证的情况下直接将其加入现有 group，但可能存在另一组基数更大的合法选择。

若显式保存全部局部 conflict matrices，临时显存为

\[
O\left(
\sum_jm_j^2
\right).
\]

实现中可采用：

1. 按 anchor 对候选池分桶并执行 batched GEMM；
2. 只保存 conflict bitset，而不保存 FP32 distance matrix；
3. 使用 squared-distance gate，避免 `sqrt`；
4. 对大候选池沿 candidate dimension 分 tile，逐 tile 构造 bitset；
5. 使用固定 degree-priority 和 stateless hash tie-break，避免保存每轮随机数；
6. 多个 anchors 的 MIS 独立并行执行。

没有候选池大小上限这一必要超参数。工程实现可以为极端大候选池设置可选计算预算，但该截断不属于 \method 的核心定义；它只会减少可接受 candidates，不会使已形成 group 失去证书保证。

certificate-based grouping core 的 per-entry 额外状态仍为 \(C_i,R_i,D_i\) 三个标量。冲突图是每轮压缩的临时工作空间，不需要跨轮保存，也不展开历史原始成员。

在线距离归一化在首轮压缩前为每层、每个 batch 和 KV head 暂时维护一个 \(d\)-维均值向量 \(\mu\)、标量 \(M_2\) 和样本数 \(n\)。每个新原始 key 的 Welford update 为 \(O(d)\)；首轮压缩后可释放 \(\mu,M_2,n\)，只保留标量 \(S_{b,h}^{\mathrm{ref}}\)。该过程不使用离线 calibration dataset。

默认 KeepKV-style selector 另外维护位置元数据 \(p\) 和 recent-score history \(P\)，其状态开销为

\[
O\left(B_{\mathrm{sz}}H_{kv}W_sL\right).
\]

当前 base-score row 可复用 attention dot-product logits；history 使用 ring buffer。更换 selector 时，\(p,P\) 可替换或省略。

### 数值实现

数学定理提供 exact-arithmetic certificate guarantee。实际 GPU 实现建议：

1. squared norm、distance GEMM、\(R_i,D_i,\widehat R_i\)、pair margins、conflict masks、certificate reductions 和 count-weighted centroid numerator 至少使用 FP32；
2. 使用
   \[
   d_{uv}^2
   =
   \max\left(
   \|K_u\|_2^2+\|K_v\|_2^2-2K_uK_v^\top,
   0
   \right)
   \]
   并比较
   \[
   d_{uv}^2
   \leq
   \left(
   \Delta-\widehat R_u-\widehat R_v
   \right)^2,
   \]
   前提是右侧 margin 非负；
3. 在阈值边界使用 conservative numerical margin，或对 \(R,D\) 做轻微向上 inflation；
4. conflict graph 的对角线固定为无 conflict；对称矩阵只需计算或存储一个三角部分；
5. priority 按低 conflict degree、低 anchor certificate、稳定 hash 的字典序确定；conflict neighbors 不得在同一轮同时获胜；
6. \(C_i\) 至少使用 INT32，极长上下文使用 INT64；\(\log C_i\) 和 \(C_iK_i,C_iV_i\) 使用 FP32；
7. selector history 和 Welford statistics 至少采用 FP32 accumulation。

由于 retained anchor 始终满足

\[
C_j^{(r)}\geq1,
\]

所以 masked count-weighted reduction 不会产生零分母。有限精度实现中的安全性依赖上述保守处理，不应将精确实数运算下的零 violation 无条件扩展到任意浮点实现。

---

# 附录 A：理论保证

## A.1 成员集合与证书不变量

初始时，

\[
\mathcal S_i^{(0)}=\{i\}.
\]

对于 retained anchor \(j\)，成员集合按照

\[
\mathcal S_j^{(r+1)}
=
\mathcal S_j^{(r)}
\uplus
\biguplus_{i\in\mathcal A_j^{(r)}}
\mathcal S_i^{(r)}
\]

更新。由于当前不同 entries 的成员集合两两不交、retained/evicted sets 不交，且每个 evicted entry 至多属于一个 \(\mathcal A_j^{(r)}\)，更新后各当前 entry 的成员集合仍两两不交。

由 Step 5 的 count-weighted reduction，可归纳得到

\[
C_i^{(r)}
=
\left|\mathcal S_i^{(r)}\right|,
\]

\[
C_i^{(r)}K_i^{(r)}
=
\sum_{x\in\mathcal S_i^{(r)}}k_x,
\qquad
C_i^{(r)}V_i^{(r)}
=
\sum_{x\in\mathcal S_i^{(r)}}v_x.
\]

定义真实 centroid radius 与真实 diameter：

\[
r_i^\star
=
\max_{x\in\mathcal S_i^{(r)}}
\left\|k_x-K_i^{(r)}\right\|_2,
\]

\[
d_i^\star
=
\max_{x,y\in\mathcal S_i^{(r)}}
\left\|k_x-k_y\right\|_2.
\]

证书不变量为

\[
R_i^{(r)}
\geq
r_i^\star,
\qquad
D_i^{(r)}
\geq
d_i^\star.
\]

由于

\[
K_i^{(r)}
=
\frac{1}{C_i^{(r)}}
\sum_{y\in\mathcal S_i^{(r)}}k_y,
\]

对任意 \(x\in\mathcal S_i^{(r)}\)，

\[
\begin{aligned}
\left\|k_x-K_i^{(r)}\right\|_2
&=
\left\|
\frac{1}{C_i^{(r)}}
\sum_{y\in\mathcal S_i^{(r)}}(k_x-k_y)
\right\|_2\\
&\leq
\frac{1}{C_i^{(r)}}
\sum_{y\neq x}
\left\|k_x-k_y\right\|_2\\
&\leq
\frac{C_i^{(r)}-1}{C_i^{(r)}}
d_i^\star\\
&\leq
\frac{C_i^{(r)}-1}{C_i^{(r)}}
D_i^{(r)}.
\end{aligned}
\]

因此

\[
\widehat R_i^{(r)}
=
\min\left(
R_i^{(r)},
\frac{C_i^{(r)}-1}{C_i^{(r)}}D_i^{(r)}
\right)
\]

仍满足

\[
\widehat R_i^{(r)}
\geq
r_i^\star.
\]

新 token 的 \(R_i=D_i=0\)，所以初始证书不变量成立。

---

## A.2 定理 1：基于冲突图 MIS 的完整组直径保证

**定理 1。** 对 retained anchor \(j\)，令

\[
\mathcal Q_j^{(r)}
=
\operatorname{ParallelMIS}
\left(
\mathcal H_j^{(r)}
\right),
\]

且最终 accepted entry group 为

\[
\overline G_j^{(r)}
=
\{j\}
\cup
\mathcal Q_j^{(r)}.
\]

则该组所代表的原始 token 集合

\[
\mathcal S_j^{(r+1)}
=
\mathcal S_j^{(r)}
\uplus
\biguplus_{i\in\mathcal Q_j^{(r)}}
\mathcal S_i^{(r)}
\]

满足

\[
\boxed{
\max_{x,y\in\mathcal S_j^{(r+1)}}
\left\|k_x-k_y\right\|_2
\leq
\Delta_{b,h}^{\mathrm{ref}}.
}
\]

因此

\[
\overline{\operatorname{diam}}
\left(
\mathcal S_j^{(r+1)}
\right)
\leq
\lambda.
\]

**证明。**

记

\[
\Delta
=
\Delta_{b,h}^{\mathrm{ref}}.
\]

任取新组中的两个原始 keys，按其来源分为四种情况。

### 情况 1：二者均来自 retained anchor \(j\)

所有进入候选池的 candidates 都要求 anchor 通过内部直径认证。若本轮发生合并，则

\[
D_j^{(r)}
\leq
\Delta.
\]

因此

\[
\left\|k_x-k_y\right\|_2
\leq
D_j^{(r)}
\leq
\Delta.
\]

### 情况 2：二者均来自同一个 accepted candidate \(i\)

由 Step 2 的 anchor compatibility，

\[
\Gamma_{ij}^{(r)}
\leq
\Delta.
\]

而

\[
\Gamma_{ij}^{(r)}
\geq
D_i^{(r)}.
\]

所以

\[
\left\|k_x-k_y\right\|_2
\leq
D_i^{(r)}
\leq
\Delta.
\]

### 情况 3：一个来自 anchor \(j\)，另一个来自 accepted candidate \(i\)

由于 \(i\in\mathcal P_j^{(r)}\)，有

\[
\Gamma_{ij}^{(r)}
\leq
\Delta.
\]

因此

\[
\begin{aligned}
\left\|k_x-k_y\right\|_2
&\leq
\widehat R_j^{(r)}
+
d_{ij}^{(r)}
+
\widehat R_i^{(r)}\\
&=
B_{ij}^{(r)}\\
&\leq
\Gamma_{ij}^{(r)}\\
&\leq
\Delta.
\end{aligned}
\]

### 情况 4：二者来自两个不同 accepted candidates \(i\neq\ell\)

\(\mathcal Q_j^{(r)}\) 是 conflict graph \(\mathcal H_j^{(r)}\) 的 independent set，因此 selected vertices \(i,\ell\) 之间不存在 conflict edge：

\[
(i,\ell)
\notin
\mathcal E_j^{(r)}.
\]

根据 conflict edge 的定义，

\[
\Gamma_{i\ell}^{(r)}
\leq
\Delta.
\]

于是

\[
\begin{aligned}
\left\|k_x-k_y\right\|_2
&\leq
\widehat R_i^{(r)}
+
d_{i\ell}^{(r)}
+
\widehat R_\ell^{(r)}\\
&=
B_{i\ell}^{(r)}\\
&\leq
\Gamma_{i\ell}^{(r)}\\
&\leq
\Delta.
\end{aligned}
\]

四种情况覆盖新组中的任意原始 key pair。证毕。

**说明。** 上述安全性只依赖 \(\mathcal Q_j^{(r)}\) 是 independent set，不依赖其是否 maximum 或 maximal。因此，即使 MIS priority、tie-break hash或并行调度改变了最终选择，已接受 group 的完整直径保证仍保持不变。maximal 性质仅保证当前结果不能直接再加入一个被排除 candidate。

---

## A.3 定理 2：最终 merged key 偏差保证

**定理 2。** 在定理 1 的条件下，对于新合并组中的任意原始 key \(k_x\)，最终 merged key 满足

\[
\left\|k_x-K_j^{(r+1)}\right\|_2
\leq
\frac{C_j^{(r+1)}-1}{C_j^{(r+1)}}
\Delta_{b,h}^{\mathrm{ref}}
\leq
\Delta_{b,h}^{\mathrm{ref}}.
\]

**证明。**

\[
K_j^{(r+1)}
=
\frac{1}{C_j^{(r+1)}}
\sum_{y\in\mathcal S_j^{(r+1)}}k_y.
\]

因此

\[
\begin{aligned}
\left\|k_x-K_j^{(r+1)}\right\|_2
&=
\left\|
\frac{1}{C_j^{(r+1)}}
\sum_y(k_x-k_y)
\right\|_2\\
&\leq
\frac{1}{C_j^{(r+1)}}
\sum_{y\neq x}
\left\|k_x-k_y\right\|_2\\
&\leq
\frac{C_j^{(r+1)}-1}{C_j^{(r+1)}}
\Delta_{b,h}^{\mathrm{ref}}.
\end{aligned}
\]

证毕。

---

## A.4 定理 3：跨轮次证书传播

**定理 3。** Step 6 的更新具有两项性质。

### 无条件证书有效性

无论本轮是否接受 candidate，更新后的证书始终满足

\[
D_j^{(r+1)}
\geq
\max_{x,y\in\mathcal S_j^{(r+1)}}
\left\|k_x-k_y\right\|_2,
\]

\[
R_j^{(r+1)}
\geq
\max_{x\in\mathcal S_j^{(r+1)}}
\left\|k_x-K_j^{(r+1)}\right\|_2.
\]

### 新合并组的阈值保证

若

\[
\mathcal Q_j^{(r)}
\neq\varnothing,
\]

则

\[
D_j^{(r+1)}
\leq
\Delta_{b,h}^{\mathrm{ref}}.
\]

**证明。**

若没有 accepted candidate，entry 仅继承原组，

\[
D_j^{(r+1)}
=
D_j^{(r)}.
\]

A.1 已证明 \(\widehat R_j^{(r)}\) 是有效半径证书，所以保存

\[
R_j^{(r+1)}
=
\widehat R_j^{(r)}
\]

保持不变量。

若存在 accepted candidates，Step 6 定义

\[
D_j^{(r+1)}
=
\max\left\{
\max_{u\in\overline G_j^{(r)}}D_u^{(r)},
\;
\max_{\substack{u,v\in\overline G_j^{(r)}\\u<v}}
B_{uv}^{(r)}
\right\}.
\]

若两个原始 keys 来自同一旧 entry \(u\)，其距离由 \(D_u^{(r)}\) 上界；若来自两个不同旧 entries \(u,v\)，其距离由

\[
B_{uv}^{(r)}
=
\widehat R_u^{(r)}
+
d_{uv}^{(r)}
+
\widehat R_v^{(r)}
\]

上界。因此 \(D_j^{(r+1)}\) 是真实组直径的有效证书。

根据定理 1，所有 selected entries 的内部直径和 cross-entry bounds 均不超过

\[
\Delta_{b,h}^{\mathrm{ref}},
\]

所以

\[
D_j^{(r+1)}
\leq
\Delta_{b,h}^{\mathrm{ref}}.
\]

对于半径证书，若原始 key 来自旧 entry \(u\)，则其到新 centroid 的距离不超过

\[
\widehat R_u^{(r)}
+
\left\|K_u^{(r)}-K_j^{(r+1)}\right\|_2.
\]

对全部 \(u\in\overline G_j^{(r)}\) 取最大值，得到有效上界 \(\widetilde R_j^{(r+1)}\)。同时真实 centroid radius 不超过真实 diameter，所以 \(D_j^{(r+1)}\) 也是有效半径上界。两者取最小仍为有效半径证书。证毕。

---

## A.5 Attention 理论的适用假设

以下分析假设：

1. 缓存中的 \(k_x\) 是实际参与 attention dot product 的 keys；若模型采用 RoPE，则全文讨论的是已经应用 RoPE 后的 keys；
2. merged key 直接由缓存 key vectors 做 count-weighted averaging，之后不按 anchor 位置重新执行 RoPE；
3. 除
   \[
   q_tk_x^\top/\sqrt d
   \]
   外，不存在无法吸收到当前分析中的 token-specific additive logit term；
4. 以下误差界均是在给定 query \(q_t\) 及其范数的条件下成立。

---

## A.6 定理 4：任意当前缓存项的 logit-deviation 上界

**定理 4。** 对任意当前缓存项 \(j\)、任意

\[
x\in\mathcal S_j^{(r)},
\]

定义

\[
\delta_x^{(r)}(t)
=
\frac{
q_t\left(k_x-K_j^{(r)}\right)^\top
}{
\sqrt d
}.
\]

则

\[
\boxed{
\left|
\delta_x^{(r)}(t)
\right|
\leq
\frac{
\left\|q_t\right\|_2
}{
\sqrt d
}
\widehat R_j^{(r)}.
}
\]

**证明。**

由 Cauchy–Schwarz inequality 和 A.1 中的有效半径证书，

\[
\begin{aligned}
\left|
\delta_x^{(r)}(t)
\right|
&\leq
\frac{
\left\|q_t\right\|_2
\left\|k_x-K_j^{(r)}\right\|_2
}{
\sqrt d
}\\
&\leq
\frac{
\left\|q_t\right\|_2
\widehat R_j^{(r)}
}{
\sqrt d
}.
\end{aligned}
\]

证毕。

**推论 1（新合并组）。** 对本轮新形成的 certified group，

\[
\widehat R_j^{(r+1)}
\leq
\frac{
C_j^{(r+1)}-1
}{
C_j^{(r+1)}
}
\Delta_{b,h}^{\mathrm{ref}}.
\]

因此，对任意 \(x\in\mathcal S_j^{(r+1)}\)，

\[
\left|
\delta_x^{(r+1)}(t)
\right|
\leq
\frac{
\left\|q_t\right\|_2
}{
\sqrt d
}
\frac{
C_j^{(r+1)}-1
}{
C_j^{(r+1)}
}
\Delta_{b,h}^{\mathrm{ref}}.
\]

---

## A.7 定理 5：任意当前缓存项的 count-calibration residual

对任意当前缓存项 \(j\)，定义固定 query 下的 oracle group logit：

\[
\ell_j^\star(t)
=
\log
\sum_{x\in\mathcal S_j^{(r)}}
\exp\left(
\frac{q_tk_x^\top}{\sqrt d}
\right),
\]

以及实际使用的 count-calibrated group logit：

\[
\widetilde\ell_j(t)
=
\frac{
q_t{K_j^{(r)}}^\top
}{
\sqrt d
}
+
\log C_j^{(r)}.
\]

令

\[
a_j^{(r)}(t)
=
\frac{
\left\|q_t\right\|_2
}{
\sqrt d
}
\widehat R_j^{(r)}.
\]

则

\[
\boxed{
0
\leq
\ell_j^\star(t)
-
\widetilde\ell_j(t)
\leq
\frac{
\left\|q_t\right\|_2^2
\left(\widehat R_j^{(r)}\right)^2
}{
2d
}.
}
\]

**证明。**

写成

\[
\frac{q_tk_x^\top}{\sqrt d}
=
\frac{q_t{K_j^{(r)}}^\top}{\sqrt d}
+
\delta_x^{(r)}(t).
\]

由于 \(K_j^{(r)}\) 是 \(\mathcal S_j^{(r)}\) 中原始 keys 的均值，

\[
\frac{1}{C_j^{(r)}}
\sum_x
\delta_x^{(r)}(t)
=
0.
\]

将组内原始 token 看作有限集合上的均匀离散随机变量，则 \(\delta^{(r)}(t)\) 的均值为零，且由定理 4，

\[
\delta^{(r)}(t)
\in
\left[
-a_j^{(r)}(t),
a_j^{(r)}(t)
\right].
\]

因此

\[
\ell_j^\star(t)
-
\widetilde\ell_j(t)
=
\log\left(
\frac{1}{C_j^{(r)}}
\sum_x
e^{\delta_x^{(r)}(t)}
\right).
\]

Jensen inequality 给出下界 \(0\)，Hoeffding lemma 给出上界

\[
\frac{\left(a_j^{(r)}(t)\right)^2}{2}.
\]

证毕。

**推论 2（新合并组）。** 对本轮新形成的 certified group，

\[
0
\leq
\ell_j^\star(t)
-
\widetilde\ell_j(t)
\leq
\frac{
\left\|q_t\right\|_2^2
}{
2d
}
\left(
\frac{
C_j^{(r+1)}-1
}{
C_j^{(r+1)}
}
\Delta_{b,h}^{\mathrm{ref}}
\right)^2.
\]

使用较简洁的弱化形式，可写为

\[
0
\leq
\ell_j^\star(t)
-
\widetilde\ell_j(t)
\leq
\frac{
\left\|q_t\right\|_2^2
\left(\Delta_{b,h}^{\mathrm{ref}}\right)^2
}{
2d
}.
\]

因此，任意当前 compressed entry 的 residual 由其有效半径证书控制；对于新形成的 certified group，该上界进一步由当前请求固定的完整组直径预算控制。

---

## A.8 Query-dependent oracle value：定性说明

固定 query 下，组内原始 values 的 oracle attention aggregation 为

\[
v_j^\star(t)
=
\sum_xp_x(t)v_x,
\qquad
p_x(t)
=
\frac{e^{\delta_x^{(r)}(t)}}{\sum_ye^{\delta_y^{(r)}(t)}}.
\]

当 certified key group 使 \(\delta_x^{(r)}(t)\) 较小时，组内归一化 attention weights \(p_x(t)\) 接近均匀权重。因此，使用

\[
V_j^{(r)}
=
\frac{1}{C_j^{(r)}}
\sum_xv_x
\]

作为 query-independent approximation 具有合理动机。

本方法不维护 value-space radius，也不认证 value-space compactness，因此不声称无条件的 value approximation error bound。这里的逻辑仅为

\[
K\text{-space Euclidean compactness}
\Longrightarrow
\text{small within-group logit deviations}
\Longrightarrow
\text{approximately uniform group-wise weights}
\Longrightarrow
\text{count-weighted value mean approximation}.
\]

---

# 附录 B：默认 retained/evicted selector

本文默认使用 Sink-protected KeepKV-style recent-window scoring 生成 retained/evicted sets，并保留 attention sink、recent tokens 与 heavy hitters 的三段式预算。该 selector 仅是 \(\Phi\) 的默认实例，不属于 \method 的 certificate-based grouping 贡献；本文不采用 KeepKV 的 ZIP-Merging、cosine merge gate 或 attention-predicted \(K,V\) merge weights。

默认 selector 额外维护：

- 每个 entry 的位置元数据 \(p_i\)；
- recent-score history
  \[
  P^{(t)}
  \in
  \mathbb R_+^{
  B_{\mathrm{sz}}\times H_{kv}\times W_s\times L^{(t)}
  }.
  \]

其中 \(W_s\) 是 score observation window。为避免增加独立窗口超参数，默认设置

\[
W_s=W,
\]

即 score observation window 与 recent-token protection window 使用相同长度。

## B.1 Attention sinks

令

\[
W_{\mathrm{sink}}\in\mathbb N_0
\]

表示序列开头强制保留的 token 数量，默认设置

\[
W_{\mathrm{sink}}=4.
\]

根据位置元数据 \(p_i\)，定义

\[
\mathcal C_{\mathrm{sink}}^{(r)}
=
\left\{
i\in\mathcal C^{(r)}:
p_i<W_{\mathrm{sink}}
\right\}.
\]

这些 entries 在每轮均强制进入 retained set，但不进行额外隔离；只要通过正文 Step 2 的 anchor compatibility 和 Step 3 的 conflict-graph selection，它们仍可作为普通 anchors 吸收 evicted entries。

## B.2 每步 base score

对 batch \(b\)、KV head \(h\)、query step \(t\) 和当前 cache entry \(i\)，定义

\[
\boxed{
s_{b,h,t,i}
=
\frac{1}{g}
\sum_{h'\in\mathcal G(h)}
\exp\left(
\frac{
q_{b,h',t}K_{b,h,i}^{\top}
}{
\sqrt d
}
\right),
}
\]

其中 \(\mathcal G(h)\) 是共享 KV head \(h\) 的 query-head 集合，\(g=|\mathcal G(h)|\)。对 MHA，\(g=1\)；对 GQA/MQA，必须先在共享同一 KV head 的 query heads 上聚合，再对物理共享的 KV entries 做一次 retained/evicted 划分。

该 selector score 使用 cached-key base exponential score

\[
\exp(qK^\top/\sqrt d),
\]

而不是 softmax 后的 attention probability，也不加入 \(+\log C_i\)。count 已由独立状态 \(C_i\) 表示；若 selector score 再包含 \(+\log C_i\)，大 group 会因 count calibration 自动获得更高 retained 概率，可能形成长期 lock-in。对 causal mask 下不可见的 query–entry pair，score 置为零。

## B.3 Recent-score buffer

\(P_{b,h,\ell,i}^{(t)}\) 表示当前 entry \(i\) 在第 \(\ell\) 个近期 query 下的 base score。在每个新 query step：

1. 为新生成的 cache entry 在已有历史 query rows 中追加零列；
2. 将当前 query 的 base score row \(s_{t,:}\) 写入 history；
3. 只保留最近 \(W_s\) 行。

抽象写为

\[
P^{(t)}
=
\operatorname{Tail}_{W_s}
\left(
\operatorname{AppendRow}
\left(
\operatorname{AppendZeroColumn}(P^{(t-1)}),
s_t
\right)
\right).
\]

新 token 在产生之前的历史 query rows 中没有合法 score，因此对应值为零。实现时使用长度为 \(W_s\) 的 circular/ring buffer，避免每步移动或拼接完整 history tensor。

## B.4 Recent-window importance

对最近 \(W_s\) 个 query rows 使用线性时间权重：

\[
\omega_\ell
=
1+
\frac{\ell-1}{W_s-1}(\beta-1),
\qquad
\ell=1,\ldots,W_s,
\]

其中越新的 query 对应越大的 \(\ell\)，且 \(\beta\geq1\) 是 selector 配置。当 \(W_s=1\) 时，定义 \(\omega_1=1\)。entry \(i\) 的 recent-window importance 为

\[
\boxed{
J_i^{(t)}
=
\sum_{\ell=1}^{W_s}
\omega_\ell P_{\ell,i}^{(t)}.
}
\]

为保持第一版简单，默认实验可固定 \(\beta=1\)，即使用未加权 recent-window sum；\(\beta>1\) 的 recency weighting 作为 selector 消融或由 validation 决定。本文不声称某个 \(\beta\) 数值是 KeepKV 的统一默认配置，也不将该 scoring 机制作为贡献。

## B.5 Retained/evicted 划分

设目标 retained budget 为 \(N_{\mathrm{keep}}^{(r)}\)。从非 sink entries 中选择最近的 tokens：

\[
N_{\mathrm{rec}}^{(r)}
=
\min\left(
W,
\max\left(
0,
N_{\mathrm{keep}}^{(r)}
-
\left|\mathcal C_{\mathrm{sink}}^{(r)}\right|
\right)
\right),
\]

\[
\mathcal C_{\mathrm{recent}}^{(r)}
=
\operatorname{Recent}
\left(
\mathcal C^{(r)}
\setminus
\mathcal C_{\mathrm{sink}}^{(r)},
N_{\mathrm{rec}}^{(r)}
\right).
\]

剩余 heavy-hitter budget 为

\[
N_{\mathrm{HH}}^{(r)}
=
N_{\mathrm{keep}}^{(r)}
-
\left|
\mathcal C_{\mathrm{sink}}^{(r)}
\cup
\mathcal C_{\mathrm{recent}}^{(r)}
\right|.
\]

定义

\[
\mathcal C_{\mathrm{old}}^{(r)}
=
\mathcal C^{(r)}
\setminus
\left(
\mathcal C_{\mathrm{sink}}^{(r)}
\cup
\mathcal C_{\mathrm{recent}}^{(r)}
\right).
\]

从 old set 中按照 recent-window importance 选择 heavy hitters：

\[
\boxed{
\mathcal C_{\mathrm{HH}}^{(r)}
=
\operatorname{TopK}
\left(
\left\{
J_i^{(r)}:
i\in\mathcal C_{\mathrm{old}}^{(r)}
\right\},
N_{\mathrm{HH}}^{(r)}
\right).
}
\]

最终

\[
\boxed{
\mathcal C_{\mathrm{keep}}^{(r)}
=
\mathcal C_{\mathrm{sink}}^{(r)}
\cup
\mathcal C_{\mathrm{recent}}^{(r)}
\cup
\mathcal C_{\mathrm{HH}}^{(r)},
}
\]

\[
\boxed{
\mathcal C_{\mathrm{evict}}^{(r)}
=
\mathcal C^{(r)}
\setminus
\mathcal C_{\mathrm{keep}}^{(r)}.
}
\]

有效配置需要满足

\[
N_{\mathrm{keep}}^{(r)}
\geq
\left|\mathcal C_{\mathrm{sink}}^{(r)}\right|.
\]

与全历史累计注意力不同，进入 heavy-hitter 比较的 old entries 使用相同长度的近期 query observation window，从而减弱仅由 token 年龄和累计观测次数产生的系统性优势。H2O-style raw cumulative attention 可作为 selector baseline，但不再作为本文默认实例。

## B.6 合并后的 selector history

历史张量 \(P_i^{(r)}\in\mathbb R_+^{W_s}\) 是 entry-level selector state。正文生成 membership mask 后，使用与 \(K,V\) 相同的 membership mask，对已有 score-history rows 独立执行 count-weighted reduction：

\[
\boxed{
P_j^{(r+1)}
=
\frac{
C_j^{(r)}P_j^{(r)}
+
\sum_{
i\in\mathcal C_{\mathrm{evict}}^{(r)}
}
M_{ij}^{(r)}C_i^{(r)}P_i^{(r)}
}{
C_j^{(r+1)}
}.
}
\]

该更新使合并时已有 history rows 保持 represented original members 的 count-weighted mean 语义。后续 query rows 则使用当前 merged centroid 的 base score 继续写入 ring buffer。未通过认证并最终丢弃的 evicted entries，其 \(P_i\) 同时删除。

\(P\) 只用于 selector，不参与 \(K,V\) 的 merge weight，也不参与 \(R,D\) certificate。\(K,V\) 仍严格使用 count-weighted arithmetic means，以保持原始成员均值、半径收紧、merged-centroid deviation、Hoeffding 零均值条件和 \(+\log C\) residual 理论。

每个 merged entry 继承 retained anchor 的位置元数据：

\[
p_j^{(r+1)}=p_j^{(r)},
\]

不对位置索引做平均。被合并 candidates 的原位置不再参与后续 recent 判断；若 anchor 属于 sink set，其 sink 身份继续保留。该位置元数据只服务于 selector，不对 cached keys 重新执行位置变换。

主方法仍允许 sink entries 作为普通 anchors。实验中应补充“sink 可吸收 candidates”与“sink 禁止吸收 candidates”的消融，但当前版本不据此改变默认方法。
