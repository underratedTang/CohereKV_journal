# \method：动机、方法流程与理论保证

## 1. 动机

KV cache eviction 能够直接控制缓存容量，但会永久删除低历史重要性 token 的 KV 信息；分组合并则可将这些待删除信息尽可能转移到 retained KV 中，并把多个指向同一 retained token 的合并操作转化为一次独立的组归约，因此同时具有信息保留价值和并行执行价值。其核心困难是：最终一个 merged key 需要代表整个多 token 组，理论上要求组内每个原始 key 相对最终 merged key 的偏差足够小，但未来 query 未知，直接执行组内两两比较或迭代聚类又会带来较高开销和顺序依赖。现有局部方法通过相邻位置、固定 chunk 或语义边界缩小搜索范围，可能遗漏跨边界但真正相似的 keys；大范围独立匹配方法可以并行地把 evicted entries 路由到 retained entries，却只检查局部的 entry-to-anchor 关系，不能直接保证同组所有原始 keys 的整体紧凑性。\method 因此将 retained/evicted 集合的产生与后续分组解耦：任意重要性策略均可先给出两个集合，\method 再将最近 retained 匹配仅作为候选生成，并利用可跨压缩轮次传播的欧氏半径与直径证书和 Boolean mask，在不展开历史成员、不执行组内全配对比较和迭代聚类的情况下，构造满足明确组紧凑度约束、可通过固定阶段并行归约的多 token groups。

---

## 2. 方法目标与基本符号

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

\((K,V,C,R,D)\) 是 certificate-based grouping core 的必要状态。本文采用的默认 selector 另外维护位置元数据 \(p\) 和 recent-score history \(P\)；替换 selector 时，\(p,P\) 可相应替换或省略。

\method 的目标是在不显式访问 \(\mathcal S_i^{(r)}\) 的情况下，利用每个缓存项的 centroid、有效半径证书和直径证书，构造任意原始成员两两欧氏距离均受控的 groups。

---

## 3. 归一化组紧凑度

对于一个由原始 keys 构成的组 \(G\)，定义其 key-space Euclidean diameter：

\[
\operatorname{diam}(G)
=
\max_{x,y\in G}
\left\|k_x-k_y\right\|_2.
\]

对于 batch \(b\)、KV head \(h\)，定义当前 compressed entries 的 count-weighted centroid RMS scale：

\[
\boxed{
S_{b,h}^{(r)}
=
\sqrt{
\frac{
\sum_{u=1}^{L^{(r)}}
C_{b,h,u}^{(r)}
\left\|K_{b,h,u}^{(r)}\right\|_2^2
}{
\sum_{u=1}^{L^{(r)}}
C_{b,h,u}^{(r)}
}
}
+\epsilon_s,
}
\]

其中 \(\epsilon_s>0\) 仅用于数值稳定。该尺度考虑了不同 compressed entries 所代表的 token 数量，但由于历史组内二阶方差未被保存，它不是全部原始 keys 的精确 RMS。

该尺度基于当前轮次的 compressed centroids，因此

\[
\Delta_{b,h}^{(r)}
=
(1-\tau)S_{b,h}^{(r)}
\]

是 round-relative diameter budget。不同轮次的 compactness 数值不应被解释为完全相同绝对尺度下的直接比较；实验中应记录 \(S_{b,h}^{(r)}\) 与 \(\Delta_{b,h}^{(r)}\) 随压缩轮次的变化。

定义归一化组紧凑度

\[
\operatorname{Comp}_{\mathrm{group}}(G)
=
\max\left(
0,\,
1-\frac{\operatorname{diam}(G)}{S_{b,h}^{(r)}}
\right).
\]

在 certificate-based grouping 模块中，控制目标组直径的唯一主要质量超参数为

\[
\tau\in(0,1).
\]

每轮允许的最大组直径为

\[
\boxed{
\Delta_{b,h}^{(r)}
=
(1-\tau)S_{b,h}^{(r)}.
}
\]

对给定阈值 \(\Delta\)，若一个 group 所代表的全部原始 keys 满足

\[
\max_{x,y\in G}
\left\|k_x-k_y\right\|_2
\leq
\Delta,
\]

则称该 group 满足 \(\Delta\)-complete consistency。等价地，经过认证的新合并组满足

\[
\operatorname{Comp}_{\mathrm{group}}(G)\geq\tau.
\]

\(\tau\) 越大，允许的完整组欧氏直径越小，分组越严格。例如：

- \(\tau=0.9\)：目标组直径为当前尺度的 \(10\%\)；
- \(\tau=0.8\)：为 \(20\%\)；
- \(\tau=0.7\)：为 \(30\%\)；
- \(\tau=0.6\)：为 \(40\%\)。

数学定义域为

\[
\tau\in(0,1).
\]

初始实验可测试

\[
\tau\in\{0.6,0.7,0.8,0.9\},
\]

并暂以

\[
\tau=0.7
\]

作为初始配置。该数值不是理论推荐值；不同模型、层、KV head 和 key-norm distribution 的合适设置应由 validation experiments 决定。

---

## 4. \method 完整流程

### Step 0：设置组紧凑度阈值

给定

\[
\tau\in(0,1),
\]

先计算当前 batch、KV head 的 count-weighted centroid RMS scale：

\[
S_{b,h}^{(r)}
=
\sqrt{
\frac{
\sum_u C_{b,h,u}^{(r)}
\left\|K_{b,h,u}^{(r)}\right\|_2^2
}{
\sum_u C_{b,h,u}^{(r)}
}
}
+\epsilon_s.
\]

随后得到本轮最大允许完整组直径

\[
\Delta_{b,h}^{(r)}
=
(1-\tau)S_{b,h}^{(r)}.
\]

\(\tau\) 在此转换为欧氏直径预算，并在 Step 3 的 pair-wise feasibility mask 中使用。较大的 \(\tau\) 产生较小的 \(\Delta_{b,h}^{(r)}\)，因而接受更少但更紧凑的 groups；较小的 \(\tau\) 放宽完整组直径和后续 approximation bounds。

### Step 1：由外部重要性策略划分 retained/evicted 集合

\method 不限定重要性分数的具体形式。给定当前缓存集合

\[
\mathcal C^{(r)}
=
\left\{
\left(
K_i^{(r)},V_i^{(r)},C_i^{(r)},R_i^{(r)},D_i^{(r)}
\right)
\right\}_{i=1}^{L^{(r)}},
\]

任意重要性选择器 \(\Phi\) 只需输出 retained set 和 evicted set：

\[
\left(
\mathcal C_{\mathrm{keep}}^{(r)},
\mathcal C_{\mathrm{evict}}^{(r)}
\right)
=
\Phi\left(
\mathcal C^{(r)},N_{\mathrm{keep}}^{(r)}
\right),
\]

并满足

\[
\mathcal C_{\mathrm{keep}}^{(r)}
\cap
\mathcal C_{\mathrm{evict}}^{(r)}
=
\varnothing,
\]

\[
\mathcal C_{\mathrm{keep}}^{(r)}
\cup
\mathcal C_{\mathrm{evict}}^{(r)}
=
\mathcal C^{(r)},
\]

\[
\left|
\mathcal C_{\mathrm{keep}}^{(r)}
\right|
=
N_{\mathrm{keep}}^{(r)}.
\]

因此，\method 可以适配累计注意力、当前注意力、位置启发式、学习得到的重要性、随机选择或其他 eviction 策略。后续分组、认证和合并仅依赖 \(\mathcal C_{\mathrm{keep}}^{(r)}\) 与 \(\mathcal C_{\mathrm{evict}}^{(r)}\)，不依赖选择器内部如何计算重要性。

#### 本文采用的默认实例：Sink-protected KeepKV-style recent-window 选择

为给出完整、可复现的默认配置，本文采用成熟的 KeepKV-style recent-window scoring 逻辑生成 retained/evicted sets，并保留 attention sink、recent tokens 与 heavy hitters 的三段式预算。该 selector 仅是 \(\Phi\) 的默认实例，不属于 \method 的 certificate-based grouping 贡献；本文不采用 KeepKV 的 ZIP-Merging、cosine merge gate 或 attention-predicted \(K,V\) merge weights。

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

##### Attention sinks

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

这些 entries 在每轮均强制进入 retained set，但不进行额外隔离；只要通过 Step 3 的 feasibility certification，它们仍可作为普通 anchors 吸收 evicted entries。

##### 每步 base score

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

##### Recent-score buffer

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

##### Recent-window importance

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

##### Retained/evicted 划分

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

##### 合并后的 selector history

历史张量 \(P_i^{(r)}\in\mathbb R_+^{W_s}\) 是 entry-level selector state。发生 certified merge 后，使用与 \(K,V\) 相同的 membership mask，对已有 score-history rows 独立执行 count-weighted reduction：

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

### Step 2：并行计算 evicted–retained 欧氏距离

对于 evicted entry \(i\) 和 retained entry \(j\)，定义 cached-key Euclidean distance

\[
d_{ij}^{(r)}
=
\left\|K_i^{(r)}-K_j^{(r)}\right\|_2.
\]

其平方可写为

\[
\left(d_{ij}^{(r)}\right)^2
=
\left\|K_i^{(r)}\right\|_2^2
+
\left\|K_j^{(r)}\right\|_2^2
-
2K_i^{(r)}{K_j^{(r)}}^\top.
\]

因此，全部

\[
\left|
\mathcal C_{\mathrm{evict}}^{(r)}
\right|
\times
\left|
\mathcal C_{\mathrm{keep}}^{(r)}
\right|
\]

距离可以通过一次 GEMM 和逐元素运算并行得到。浮点实现中使用

\[
d_{ij}^{(r)}
=
\sqrt{
\max\left(
\left(d_{ij}^{(r)}\right)^2,0
\right)
}
\]

避免舍入误差产生微小负平方距离。

---

### Step 3：最近可行 anchor 与完整组 Boolean mask

Step 0 给出的

\[
\Delta_{b,h}^{(r)}
=
(1-\tau)S_{b,h}^{(r)}
\]

是本轮新合并组的最大允许欧氏直径。

对于任意 evicted–retained pair \((i,j)\)，定义 candidate entry \(i\) 相对 retained center \(K_j^{(r)}\) 的覆盖半径上界：

\[
E_{ij}^{(r)}
=
\widehat R_i^{(r)}
+
d_{ij}^{(r)}.
\]

定义 anchor \(j\) 可接受的 candidate 覆盖半径预算：

\[
\boxed{
T_j^{(r)}
=
\max\left\{
0,\,
\min\left(
\frac{\Delta_{b,h}^{(r)}}{2},
\Delta_{b,h}^{(r)}-\widehat R_j^{(r)}
\right)
\right\}.
}
\]

其中：

- \(E_{ij}^{(r)}\leq\Delta_{b,h}^{(r)}-\widehat R_j^{(r)}\) 控制 anchor–candidate 的任意原始成员距离；
- \(E_{ij}^{(r)}\leq\Delta_{b,h}^{(r)}/2\) 控制两个不同 accepted candidates 之间的任意原始成员距离。
- 当 \(T_j^{(r)}=0\) 时，anchor \(j\) 不再具有正的吸收预算；仅 \(E_{ij}^{(r)}=0\) 的 candidate 仍可能通过。

先对所有 evicted–retained pairs 构造可行性 mask：

\[
\boxed{
F_{ij}^{(r)}
=
\mathbb I
\left[
D_j^{(r)}
\leq
\Delta_{b,h}^{(r)}
\right]
\cdot
\mathbb I
\left[
D_i^{(r)}
\leq
\Delta_{b,h}^{(r)}
\right]
\cdot
\mathbb I
\left[
E_{ij}^{(r)}
\leq
T_j^{(r)}
\right].
}
\]

\(D_i^{(r)}\leq\Delta_{b,h}^{(r)}\) 是一个偏保守但便于跨轮次传播的内部一致性条件。由于 \(\Delta_{b,h}^{(r)}\) 每轮可能随尺度变化，上轮合法的 entry 不一定自动满足本轮阈值，因此两个直径条件不能无条件省略。

每个 evicted entry 在所有可行 retained anchors 中选择 centroid 距离最近者：

\[
j_i^\star
=
\min\left(
\operatorname*{arg\,min}_{
j:
F_{ij}^{(r)}=1
}
d_{ij}^{(r)}
\right).
\]

前置的 \(\min\) 表示当多个 anchors 距离相同时，选择当前 retained tensor 中稳定局部索引最小者，以保证确定性；该索引不等同于原始 token position。若对所有 \(j\) 都有

\[
F_{ij}^{(r)}=0,
\]

则 \(j_i^\star\) 不定义，candidate \(i\) 被丢弃。

最终 membership mask 为

\[
\boxed{
M_{ij}^{(r)}
=
F_{ij}^{(r)}
\cdot
\mathbb I
\left[
j=j_i^\star
\right].
}
\]

因此

\[
\sum_jM_{ij}^{(r)}
\leq1.
\]

这种“先认证、再选择最近可行 anchor”的顺序避免了以下情况：绝对最近 anchor 因历史半径过大而不可行，但另一个稍远 anchor 实际满足完整组证书，candidate 却被不必要地丢弃。该修改不引入迭代、顺序依赖或组内 candidate–candidate all-pairs comparison。

若默认 selector 中的 attention-sink entry 满足 \(F_{ij}^{(r)}=1\) 且距离最近，它同样可以成为最终 anchor。

#### 与普通星形 gate 的区别

普通星形 gate 只要求

\[
d_{ij}^{(r)}
\leq
\Delta_{b,h}^{(r)}.
\]

即使 \(B\) 和 \(C\) 都通过，只能推出

\[
\left\|k_B-k_C\right\|_2
\leq
2\Delta_{b,h}^{(r)},
\]

不能保证目标距离 \(\Delta_{b,h}^{(r)}\)。

\method 的星形结构只用于组织 evicted–retained pairs；最终 membership 由 entry 内部直径、anchor–candidate 距离和 candidate–candidate 距离共同认证。因此，

\[
M_{ij}^{(r)}
=
M_{\ell j}^{(r)}
=
1
\]

可推出新合并组中的对应原始成员两两距离均不超过 \(\Delta_{b,h}^{(r)}\)，而无需显式计算同组 candidates 之间的 all-pairs 距离。证明见附录 A.2。

---

### Step 4：形成最终 groups 并更新成员集合

对每个 retained entry \(j\)，定义参与本轮 reduction 的 compressed-entry group：

\[
G_j^{(r)}
=
\{j\}
\cup
\mathcal A_j^{(r)},
\]

其中

\[
\mathcal A_j^{(r)}
=
\left\{
i\in\mathcal C_{\mathrm{evict}}^{(r)}
:
M_{ij}^{(r)}=1
\right\}.
\]

\(G_j^{(r)}\) 是 entry-index 层面的集合；新缓存项最终代表的原始 token 集合为

\[
\boxed{
\mathcal S_j^{(r+1)}
=
\mathcal S_j^{(r)}
\uplus
\biguplus_{i\in\mathcal A_j^{(r)}}
\mathcal S_i^{(r)},
}
\]

其中 \(\uplus\) 表示不交并。未被接受的 evicted entries 所代表的原始 token 集合随其缓存项一起从后续缓存状态中删除。

由于 retained/evicted sets 不交，且每个 evicted entry 至多被分配给一个 retained anchor，更新后的当前缓存项所代表的原始 token 集合仍两两不交。因此，

\[
C_j^{(r+1)}
=
\left|
\mathcal S_j^{(r+1)}
\right|,
\]

且 Step 5 的 count-weighted reduction 继续保持 \(K,V\) 的原始成员均值不变量。

当

\[
\mathcal A_j^{(r)}
\neq\varnothing
\]

时，\(\mathcal S_j^{(r+1)}\) 是本轮新形成的 certified original-token group。若 \(\mathcal A_j^{(r)}=\varnothing\)，entry \(j\) 仅保持原状态，本轮不重新声明其满足可能已经变化的 \(\Delta_{b,h}^{(r)}\)。

由于每个 evicted entry 至多被分配给一个 retained anchor，不同 anchors 对应的 masked segmented reductions 在数据归约层面互不重叠，因此可以并行执行。

---

### Step 5：Masked count-weighted KV reduction

更新组所代表的原始 token 数量：

\[
C_j^{(r+1)}
=
C_j^{(r)}
+
\sum_{
i\in\mathcal{C}_{\mathrm{evict}}^{(r)}
}
M_{ij}^{(r)}C_i^{(r)}.
\]

更新 key：

\[
K_j^{(r+1)}
=
\frac{
C_j^{(r)}K_j^{(r)}
+
\sum_{
i\in\mathcal{C}_{\mathrm{evict}}^{(r)}
}
M_{ij}^{(r)}C_i^{(r)}K_i^{(r)}
}{
C_j^{(r+1)}
}.
\]

更新 value：

\[
V_j^{(r+1)}
=
\frac{
C_j^{(r)}V_j^{(r)}
+
\sum_{
i\in\mathcal{C}_{\mathrm{evict}}^{(r)}
}
M_{ij}^{(r)}C_i^{(r)}V_i^{(r)}
}{
C_j^{(r+1)}
}.
\]

Boolean mask 明确区分了 candidate routing 和最终 merging：只有通过完整组认证的 entries 才会进入 count、key 和 value 的归约。

由于 \(C_iK_i\) 和 \(C_iV_i\) 分别等于 entry \(i\) 所代表的全部原始 keys 和 values 之和，上述 masked count-weighted reduction 等价于直接对所有 accepted 原始 token 求均值，且与 reduction 顺序无关。

在 certified memberships 确定后，\method 采用标准的 count-weighted centroid aggregation；该归约公式本身不是分组贡献的核心。\(K_j^{(r+1)}\) 与 \(V_j^{(r+1)}\) 分别由同一个 membership mask 独立执行 count-weighted reduction，\(V_j^{(r+1)}\) 不由 merged key 的数值计算。key-space compactness 仅用于控制组内 logit deviations，从而支持将 count-weighted value mean 作为 query-independent approximation。

默认 KeepKV-style selector 的 history \(P\) 使用同一 membership mask 做独立的 count-weighted reduction，但 \(P\) 不参与 \(K,V\) 的聚合权重计算。


---

### Step 6：更新跨轮次半径与直径证书

令

\[
\mathcal A_j^{(r)}
=
\left\{
i\in\mathcal C_{\mathrm{evict}}^{(r)}
:
M_{ij}^{(r)}=1
\right\}
\]

表示 anchor \(j\) 接受的 candidates。对于 \(i\in\mathcal A_j^{(r)}\)，沿用 Step 3 的

\[
E_{ij}^{(r)}
=
\widehat R_i^{(r)}+d_{ij}^{(r)}.
\]

令 \(E_{j,(1)}^{(r)}\) 和 \(E_{j,(2)}^{(r)}\) 分别为集合

\[
\left\{
E_{ij}^{(r)}:
i\in\mathcal A_j^{(r)}
\right\}
\]

中的最大值和第二大值，并统一规定：

- 无 accepted candidate：\(E_{j,(1)}^{(r)}=E_{j,(2)}^{(r)}=0\)；
- 仅一个 accepted candidate \(i\)：\(E_{j,(1)}^{(r)}=E_{ij}^{(r)}\)，\(E_{j,(2)}^{(r)}=0\)。

同时约定

\[
\max_{i\in\varnothing}D_i^{(r)}=0.
\]

新的直径证书定义为

\[
\boxed{
\begin{aligned}
D_j^{(r+1)}
=
\max\Bigg\{&
D_j^{(r)},\\
&
\max_{i\in\mathcal A_j^{(r)}}D_i^{(r)},\\
&
\widehat R_j^{(r)}
+
E_{j,(1)}^{(r)},\\
&
E_{j,(1)}^{(r)}
+
E_{j,(2)}^{(r)}
\Bigg\}.
\end{aligned}
}
\]

四项依次覆盖：

1. anchor entry 内部的两两距离；
2. 每个 candidate entry 内部的两两距离；
3. anchor 与任一 candidate 之间的两两距离；
4. 两个不同 candidates 之间的两两距离。

最终 merged key 从旧 centers 移动到 \(K_j^{(r+1)}\)。先计算新的半径上界

\[
\begin{aligned}
\widetilde R_j^{(r+1)}
=
\max\Bigg\{&
\widehat R_j^{(r)}
+
\left\|K_j^{(r)}-K_j^{(r+1)}\right\|_2,\\
&
\max_{i\in\mathcal A_j^{(r)}}
\left[
\widehat R_i^{(r)}
+
\left\|K_i^{(r)}-K_j^{(r+1)}\right\|_2
\right]
\Bigg\}.
\end{aligned}
\]

由于 merged centroid 位于完整组原始 keys 的凸包内，完整组直径同样是成员到 merged centroid 的有效上界，因此保存

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

该更新只需要 masked segmented-max、segmented top-2 和逐元素运算，不需要访问任何历史原始 token。

如果没有 evicted entry 通过认证，则

\[
K_j^{(r+1)}=K_j^{(r)},
\qquad
V_j^{(r+1)}=V_j^{(r)},
\]

\[
C_j^{(r+1)}=C_j^{(r)},
\qquad
D_j^{(r+1)}=D_j^{(r)},
\]

\[
R_j^{(r+1)}
=
\widehat R_j^{(r)}.
\]

---

### Step 7：Count-calibrated decoding

\method 不额外保存 bias 状态。缓存中已经维护了 count \(C^{(r+1)}\)，解码时直接计算

\[
\log C^{(r+1)}
\]

并加入 attention logits：

\[
\widetilde L_t
=
q_t{K^{(r+1)}}^\top
\cdot\frac{1}{\sqrt d}
+
\log C^{(r+1)}.
\]

实现时可将整数 count 转换、对数计算和 logit addition 融合进 QK/softmax 前处理；因此方法状态仍只有

\[
\left(
K^{(r+1)},V^{(r+1)},C^{(r+1)},R^{(r+1)},D^{(r+1)}
\right),
\]

不需要额外存储 \(B\)。

\(+\log C\) 是采用的 count-calibration 机制，而非本文声称的新 decoding 机制。\method 的理论贡献是利用 certified complete-group diameter 给出该 calibration residual 的显式上界。\(C=1\) 时自动得到零偏置。


---

## 5. 并行实现、复杂度与数值细节

\method 的一次压缩由固定阶段组成：

\[
\text{importance split}
\rightarrow
\text{distance GEMM}
\rightarrow
\text{pair-wise feasibility mask}
\rightarrow
\text{nearest feasible argmin}
\rightarrow
\text{segmented sum/max/top-2}
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
\right|.
\]

主要计算复杂度为：

- evicted–retained 距离 GEMM：\(O(N_eN_rd)\)；
- pair-wise feasibility mask：\(O(N_eN_r)\)；
- nearest feasible argmin：\(O(N_eN_r)\)；
- segmented sum/max/top-2：总归约工作量为 \(O(N_e+N_r)\) 量级。

若显式物化完整 distance matrix 和 feasibility mask，临时显存复杂度为

\[
O(N_eN_r).
\]

实现中可沿 retained dimension 分 tile，并融合 distance computation、feasibility test 与 running argmin，从而避免同时保存完整的 \(d_{ij}\) 和 \(F_{ij}\) 矩阵。

certificate-based grouping core 的额外状态为每个 entry 的 \(C_i,R_i,D_i\) 三个标量，其中 \(C_i\) 同时用于 count-weighted reduction 和 count calibration。方法仍需要 evicted–retained 的 pairwise distance matrix，但不需要同一候选组内部的 candidate–candidate all-pairs comparison，也不需要展开历史原始成员。

默认 KeepKV-style selector 另外维护位置元数据 \(p\) 和 recent-score history \(P\)，其状态开销为

\[
O\left(B_{\mathrm{sz}}H_{kv}W_sL\right).
\]

当前 base-score row 的计算可复用 attention dot-product logits；history 使用 ring buffer 后，每步写入成本为 \(O(B_{\mathrm{sz}}H_{kv}L)\)，在执行 selector TopK 时计算 weighted importance 的工作量为 \(O(B_{\mathrm{sz}}H_{kv}W_sL)\)。更换 selector 时，\(p,P\) 可替换或省略。

### 数值实现

数学定理提供的是 exact-arithmetic certificate guarantee。实际 GPU 实现建议：

1. squared norm、距离、\(R_i,D_i,E_{ij},T_j\)、segmented max/top-2 和 count-weighted centroid numerator 至少使用 FP32，即使 \(K,V\) 使用 FP16/BF16；
2. 通过
   \[
   d_{ij}
   =
   \sqrt{\max(d_{ij}^2,0)}
   \]
   计算距离；
3. 在边界比较中使用由 dtype machine precision 和当前尺度确定的微小 conservative margin，或对更新后的证书做轻微向上 inflation；
4. \(C_i\) 至少使用 INT32，极长上下文可使用 INT64；计算 \(\log C_i\) 前转换为 FP32，\(C_iK_i\) 与 \(C_iV_i\) 使用 FP32 accumulation；
5. 默认 selector 的 exponentiation 输入、recent-score history \(P\)、weighted importance \(J\)、count-weighted history numerator 和 TopK score 使用 FP32 计算或至少采用 FP32 accumulation。若显存压力较大，可单独评估用 BF16 存储 \(P\)。

由于 retained anchor 始终满足

\[
C_j^{(r)}\geq1,
\]

所以

\[
C_j^{(r+1)}
=
C_j^{(r)}
+
\sum_iM_{ij}^{(r)}C_i^{(r)}
\geq1,
\]

masked count-weighted reduction 不会产生零分母。有限精度实现中的理论安全性依赖上述保守数值处理，不应将数学上的零 violation 无条件扩展到任意浮点实现。

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

## A.2 定理 1：新合并组的完整直径保证

**定理 1。** 对于发生 certified merge 的 retained anchor \(j\)，即

\[
\mathcal A_j^{(r)}
=
\left\{
i:
M_{ij}^{(r)}=1
\right\}
\neq\varnothing,
\]

Step 3 的 feasibility mask 保证

\[
\max_{x,y\in\mathcal S_j^{(r+1)}}
\left\|k_x-k_y\right\|_2
\leq
\Delta_{b,h}^{(r)}.
\]

因此新合并组满足 \(\Delta_{b,h}^{(r)}\)-complete consistency。

若 \(\mathcal A_j^{(r)}=\varnothing\)，本轮不形成新的 merged group，entry \(j\) 仅继承已有证书；此时不重新声明其满足本轮可能变化的 \(\Delta_{b,h}^{(r)}\)。

**证明。**

记

\[
\Delta
=
\Delta_{b,h}^{(r)}.
\]

因为 \(\mathcal A_j^{(r)}\neq\varnothing\)，至少存在一个 \(i\) 满足 \(F_{ij}^{(r)}=1\)，所以

\[
D_j^{(r)}\leq\Delta.
\]

任取新组内两个原始 keys，根据来源分四种情况。

### 情况 1：二者均来自 anchor entry \(j\)

\[
\left\|k_x-k_y\right\|_2
\leq
D_j^{(r)}
\leq
\Delta.
\]

### 情况 2：二者均来自同一个 candidate entry \(i\)

由 \(F_{ij}^{(r)}=1\)，

\[
D_i^{(r)}
\leq
\Delta,
\]

所以

\[
\left\|k_x-k_y\right\|_2
\leq
\Delta.
\]

### 情况 3：一个来自 anchor \(j\)，另一个来自 candidate \(i\)

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
\widehat R_j^{(r)}
+
E_{ij}^{(r)}.
\end{aligned}
\]

由

\[
E_{ij}^{(r)}
\leq
T_j^{(r)}
\leq
\Delta-\widehat R_j^{(r)},
\]

得到

\[
\left\|k_x-k_y\right\|_2
\leq
\Delta.
\]

### 情况 4：二者来自两个不同 candidates \(i\neq \ell\)

\[
\begin{aligned}
\left\|k_x-k_y\right\|_2
&\leq
E_{ij}^{(r)}
+
E_{\ell j}^{(r)}\\
&\leq
\frac{\Delta}{2}
+
\frac{\Delta}{2}\\
&=
\Delta.
\end{aligned}
\]

四种情况覆盖新合并组中的任意原始 key pair。证毕。

---

## A.3 定理 2：最终 merged key 偏差保证

**定理 2。** 在定理 1 的条件下，对于新合并组中的任意原始 key \(k_x\)，最终 merged key 满足

\[
\left\|k_x-K_j^{(r+1)}\right\|_2
\leq
\frac{C_j^{(r+1)}-1}{C_j^{(r+1)}}
\Delta_{b,h}^{(r)}
\leq
\Delta_{b,h}^{(r)}.
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
\Delta_{b,h}^{(r)}.
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
\mathcal A_j^{(r)}
\neq\varnothing,
\]

则进一步有

\[
D_j^{(r+1)}
\leq
\Delta_{b,h}^{(r)}.
\]

**证明。**

当没有 accepted candidate 时，entry 仅继承原组，\(D_j^{(r+1)}=D_j^{(r)}\)。A.1 已证明 \(\widehat R_j^{(r)}\) 是有效半径证书，因此保存 \(R_j^{(r+1)}=\widehat R_j^{(r)}\) 仍保持不变量。

当存在 accepted candidates 时，Step 6 中直径证书的四项分别覆盖：

1. anchor 内部；
2. 每个 candidate 内部；
3. anchor–candidate；
4. 两个不同 candidates。

因此 \(D_j^{(r+1)}\) 是新组真实直径的上界。由 Step 3 的 feasibility 条件，

\[
D_j^{(r)}
\leq
\Delta,
\qquad
D_i^{(r)}
\leq
\Delta,
\]

\[
\widehat R_j^{(r)}
+
E_{j,(1)}^{(r)}
\leq
\Delta,
\]

\[
E_{j,(1)}^{(r)}
+
E_{j,(2)}^{(r)}
\leq
\Delta,
\]

所以

\[
D_j^{(r+1)}
\leq
\Delta.
\]

对半径证书，若原始 key 来自旧 anchor，则其到新 centroid 的距离不超过

\[
\widehat R_j^{(r)}
+
\left\|K_j^{(r)}-K_j^{(r+1)}\right\|_2.
\]

若来自 candidate \(i\)，则不超过

\[
\widehat R_i^{(r)}
+
\left\|K_i^{(r)}-K_j^{(r+1)}\right\|_2.
\]

故 \(\widetilde R_j^{(r+1)}\) 是有效上界。另一方面，新 centroid 位于组内 keys 的凸包中，真实 centroid radius 不超过真实 diameter，因此 \(D_j^{(r+1)}\) 也是有效半径上界。两者取最小仍是有效半径证书。证毕。

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
\Delta_{b,h}^{(r)}.
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
\Delta_{b,h}^{(r)}.
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
\Delta_{b,h}^{(r)}
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
\left(\Delta_{b,h}^{(r)}\right)^2
}{
2d
}.
\]

因此，任意当前 compressed entry 的 residual 由其有效半径证书控制；对于新形成的 certified group，该上界进一步由本轮完整组直径预算控制。

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
