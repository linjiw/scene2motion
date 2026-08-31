# Full verbatim review behind the 2026-08-31 RAMP refocus

Companion to `guidance-2026-08-31-ramp-refocus.md` (working summary + actions). This file
preserves the review as received on 2026-08-31, verbatim, so the repo no longer depends on the
conversation record. Minor paste artifacts at the head of the message (duplicated fragments of
the closing judgment) are omitted; content is otherwise unedited.

---

我认真把这版 draft 从标题、摘要、贡献、系统、实验到数据生成部分重新顺了一遍。最具体、也最重要的观察是：

> **这份稿子的工程证据已经很强，但论文身份又悄悄滑回了你明确不想写的方向。**

标题是 *A Frozen Motion Prior Is Not a Planner*；摘要的第一贡献是 audit，第三贡献是 defect catalogue；五项贡献里，真正提升能力的 method 只有第二项；§4、§6、§7、§8、§9 又都在解释"怎样测得更准、怎样拒绝、怎样复现错误、怎样生成可信数据"。所以即使开头写着 "methods + systems paper"，审稿人读完仍然很可能把它概括为：

> "一篇对 ARDY 接口进行严格评测，并加上启发式闭环修复的系统论文。"

Gemini 对它"结构扎实、叙事完整、工程可信"的判断并没有错，但它高估了这种完整性在顶会评审里的作用，也低估了一个更尖锐的问题：**当前最成熟的是测量系统，当前最薄的是能扩展到多种 traversal behavior 的新方法。**

我对这版稿子的真实判断是：

* 作为内部技术报告和研究证据库：非常强。
* 作为 artifact / evaluation-oriented systems paper：相当完整。
* 作为你真正想要的、以能力提升为核心的 CoRL/RSS/ICRA methods paper：**目前还没有完成主方法闭环。**
* 这并不是因为研究失败了，而是因为 Phase 4 已经非常准确地告诉我们下一种方法应当长什么样。

---

# 一、最难的问题不是"怎么检测碰撞"，而是如何逆向控制一个生成先验

现在 draft 把困难描述成：

* conditioning interface 不可靠；
* text 不会空间对齐；
* tracker 有执行误差；
* OOD 下 proposer 会失效。

这些都对，但它们其实是同一个更深问题的不同表现。

设冻结先验为：

$$
x^{\mathrm{ref}}_k
\sim
G_\theta(u_k,z_k,h_k),
$$

其中：

* \(u_k\) 是我们提供的 text、path、joint constraints、keyframes；
* \(z_k\) 是采样随机性；
* \(h_k\) 是自回归历史和当前 gait phase；
* \(x^{\mathrm{ref}}_k\) 是生成参考动作。

然后 SONIC 执行：

$$
x^{\mathrm{exec}}_k
=
T(x^{\mathrm{ref}}_k,s_0,\xi_k),
$$

最终我们真正关心的是：

$$
J(\mathcal S,x^{\mathrm{exec}}_k),
$$

即场景 \(\mathcal S\) 中的碰撞、通过、接触、稳定和目标到达。

传统 planner 默认 action \(u\) 的效果是可预测的。但在这里，\(u\) 不是 action，只是对生成分布的一个软条件。我们真正要解决的是：

$$
u^\star
=
\arg\min_u
\mathbb E_{z,\xi}
\left[
J\bigl(
\mathcal S,
T(G_\theta(u,z,h))
\bigr)
\right].
$$

而 \(T\circ G_\theta\)：

* 随机；
* 依赖 gait phase；
* 依赖历史窗口；
* 条件通道高度耦合；
* 在 OOD 请求下响应非线性；
* 没有可用的逆模型；
* tracker 又会进一步改变结果。

所以最难的科学问题可以浓缩成一句话：

> **如何对一个冻结的、随机的、上下文依赖的生成式运动执行器做在线系统辨识与逆向控制。**

你们现在的 audit 很有价值，是因为它暴露了这个问题；但论文真正应该贡献的是一个更好的 **response-adaptive inverse controller**，而不是把 audit 本身推到主角位置。

---

# 二、当前标题和 related-work framing 有被认为"立稻草人"的风险

ARDY 的正式定位是 streaming interactive motion generation，强调在线文本、路径和柔性运动学约束；它的训练约束来自 ground-truth poses。Kimodo 也主要把自己定位为 controllable kinematic motion generator 和 motion-authoring system，支持 full-body keyframes、稀疏位置/旋转、waypoints、paths 和 contact patterns。它们并没有直接宣称"给任意复杂场景就能成为可靠的 collision-free humanoid planner"。

真正明确提出 "motion diffusion can serve as a versatile kinematic motion planner" 的是 CLoSD 这类工作，它用 text、target location、diffusion planner 和 physics tracker 构造闭环。

因此下面这句过于宽：

> "Large frozen motion priors are increasingly proposed as drop-in planners."

审稿人可能会说：

> ARDY 和 Kimodo 主张的是 controllable generation，你们拿一个 authoring interface 去做精密 scene planning，然后证明它不是 planner。

更稳、更准确，同时也更有建设性的开场应当是：

> Large motion priors provide a powerful motion substrate for humanoid planning, but their conditioning interfaces do not directly expose a scene-grounded action space. A planner must infer how requested constraints translate into realized and physically executed whole-body motion.

换句话说，不要把相关工作写成"他们误把 prior 当 planner"，而要写成：

> **已有 prior 提供了丰富运动能力，但缺少把 scene objective 转换为 prior-compatible、可执行 motion program 的方法。**

这个 framing 不但更公平，也自然把空间留给你们的新方法。

---

# 三、为什么当前 generate–verify–repair 还不足以成为最终方法

你们当前的 repair 对 duck 轴非常合理：

$$
\Delta q(s)
=
\frac{e(s)}{|g'(s)|},
$$

再加 lag compensation、forward secant 和 monotonic deepening。作为系统组件它很不错。

但从方法审稿人的角度，它仍可能被看成：

> 一个针对 scalar crouch schedule 的反馈控制器，加上 collision checking。

这会遇到两个问题。

第一，相关工作已经把"生成后验证""controller-aware selection""test-time feedback""安全 gating"推进得很远。Texedo 用 learned grounded verifiers 对候选动作进行 controller-aware test-time selection，使用 SONIC，并报告 verifier 对未见 generator 的 plug-and-play transfer；所以"冻结 generator + verifier + best-of-\(N\) + SONIC"本身已经不足以构成新颖性。

BRIC 已经在测试时同时适配 physics controller，并在 signal space 引导 diffusion planner；ReactiveBFM 已经通过 proprioceptive feedback、短时域生成和 trajectory chunking 做 closed-loop replanning；SafeFlow 同时做 physics-guided generation 和 deployment-time safety gating。

因此 Scene2Motion 必须清楚地证明它做了这些方法没有做的事情：

> **它不是只筛选、拒绝或适配 tracker，而是根据实际生成失败，修改一个场景对齐的结构化 motion program，从而主动产生下一条更好的动作。**

第二，当前 repair 只能在已经存在的单调 duck 轴上工作。它不能自然解决：

* 哪只脚跨越；
* 抬脚最高点放在哪里；
* stance foot 是否保持；
* 侧身还是缩肩；
* 左绕还是右绕；
* 多个障碍之间如何组合动作；
* text 选中的动作语义怎样放在正确空间位置。

这就是为什么论文目前能证明"修复有效"，但还不能证明"Scene2Motion 是一个通用的 whole-body scene planner"。

---

# 四、我建议的主方法：Response-Adaptive Motion Programming

我会把项目升级为：

## **Scene2Motion: Response-Adaptive Motion Programs for Frozen Humanoid Priors**

简称可以继续用：

## **Scene2Motion-RAMP**

核心思想不是继续扩大 audit，而是把 prior 的 conditioning interface 重新组织成一种可规划、可反馈修正的 **event-aligned motion program**。

完整系统可以写成：

$$
\text{scene}
\rightarrow
\text{route events}
\rightarrow
\text{prior-compatible adaptation program}
\rightarrow
\text{frozen prior}
\rightarrow
\text{measured response}
\rightarrow
\text{program repair}
\rightarrow
\text{execution-aware selection}.
$$

这里面最关键的是两个新东西：

1. **Prior-aligned adaptation representation**
2. **Response-conditioned repair**

---

## 4.1 用 path-progress event 代替全局 frame schedule

当前 step-over 的 seed-0 gating defect 其实揭示了一个根本问题：不能把动作绑定在固定 frame 上。

对路径 \(\pi\)，将每个障碍表示为：

$$
e_j
=
\left(
s_j,\Delta s_j,g_j,b_j,m_j,\phi_j,a_j
\right),
$$

其中：

* \(s_j\)：障碍中心对应的 path progress；
* \(\Delta s_j\)：影响范围；
* \(g_j\)：局部几何描述；
* \(b_j\)：受影响身体区域；
* \(m_j\)：动作模式，例如 duck、step-over、turn-and-tuck；
* \(\phi_j\)：目标 gait/contact phase；
* \(a_j\)：连续参数，例如幅度、持续距离、躯干旋转、脚高。

例如，step-over 不是：

> 在 frame 104 把左脚提高 8 cm。

而是：

> 当机器人接近 path progress \(s_j\) 时，选择届时处于 swing phase 的脚，并让该脚的 clearance arc 在障碍中心达到峰值，同时保持另一只脚的 stance contact。

这一步直接解决：

* seed 间步态相位不同；
* 速度变化；
* obstacle placement；
* text prompt 的 window timing；
* 长 clip 自回归漂移。

---

## 4.2 不再编写孤立关节坐标，而是使用 coherent adaptation residual

ARDY 的训练约束来自真实 motion poses；Kimodo 同样是通过多种一致的 kinematic constraints 进行 motion authoring。孤立地修改一个膝盖、脚或者 pelvis coordinate，很容易形成训练中没有出现过的约束组合。

所以，下一步不应该只是测试"更密的 full-body keyframe"。那仍然是在重复测试 position channel。

更好的表示是：

$$
\Delta P
=
P_{\mathrm{adapt}}
\ominus
P_{\mathrm{nominal}},
$$

即某种 scene-induced adaptation 相对于 phase-matched normal walking 的局部残差。

例如从一个真正的 step-over motion 中提取：

* pelvis vertical residual；
* swing-foot arc residual；
* hip/knee/ankle rotation residual；
* root-speed residual；
* stance-foot contact metadata；
* torso and arm counterbalance；
* event timing residual。

然后对当前 seed 生成的 nominal walk：

$$
P_{\mathrm{target}}^{z}
=
P_{\mathrm{nominal}}^{z}
\oplus
W_{\gamma,\tau,\rho}
(\Delta P),
$$

其中：

* \(\gamma\)：动作幅度；
* \(\tau\)：沿路径的空间平移；
* \(\rho\)：根据速度和障碍宽度进行时间缩放。

这不是"复制一个绝对 donor pose"。它保留当前 seed 自己的：

* gait style；
* swing/stance foot；
* root velocity；
* heading；
* motion history。

你们自己的 MTC 数据恰好可以为这一步提供 scene-induced adaptations。MTC 包含 348 条轨迹和 145 个场景，并且任务本身强调 lateral clearance、height adaptation 和 asymmetric whole-body adjustment，这比继续人工写 step-over coordinates 更接近真正的监督信号。

MotionBricks 已经提出 smart primitives 和 modular motion authoring，因此"使用 motion primitives"本身不能作为新颖性；你们真正的新意必须是：

> **从 scene-aligned adaptation residual 构造事件程序，并依据 prior 实际响应和物理执行结果进行修正。**

---

## 4.3 学习的对象不应再是 open-loop schedule，而应是 response-conditioned correction

当前 TCN 做的是：

$$
\hat u_0
=
f_\phi(\mathcal S),
$$

或者预测某个场景下应该使用什么 schedule。

但它没有看到当前这一次 ARDY 究竟生成了什么。于是它必须凭 scene 输入猜测：

* latent sample；
* 当前 gait phase；
* prompt response；
* autoregressive history；
* constraint interaction；
* generator-specific deformation。

这几乎是在学习一个全局 ARDY simulator，所以很容易 distribution shift。

新的 RepairNet 应当输入：

$$
r_k
=
\Phi
\left(
\mathcal S,
u_k,
x_k^{\mathrm{ref}},
x_k^{\mathrm{exec}}
\right),
$$

其中 \(r_k\) 包括：

* signed clearance trace；
* collision body；
* collision path position；
* achieved pelvis dip；
* body width；
* swing-foot peak position；
* stance-contact loss；
* root deviation；
* execution clearance loss；
* previous repair history。

然后预测：

$$
\Delta u_k
=
\pi_\phi
(\mathcal S,u_k,r_k),
$$

并更新：

$$
u_{k+1}
=
\Pi_{\mathcal U}
(u_k+\Delta u_k).
$$

这背后的逻辑是：**不再让网络猜 prior 会怎么响应，而是先观察它怎么响应，再学会怎么修。**

最有说服力的 ablation 将是：

| 模型                                           | Scene | 当前 program | 实际生成 response |
| -------------------------------------------- | ----: | ---------: | ------------: |
| 当前 TCN proposer                              |     ✓ |            |               |
| Program-conditioned proposer                 |     ✓ |          ✓ |               |
| Response-conditioned RepairNet               |     ✓ |          ✓ |             ✓ |
| RepairNet without coherent packet projection |     ✓ |          ✓ |             ✓ |

如果第三个显著优于前两个，你们就拥有一个真正清晰的方法结论：

> **Closed-loop learning should occur in response space, not only in scene-to-command space.**

---

## 4.4 先用 optimizer 当 teacher，再把它蒸馏成 RepairNet

不要一开始就直接训练一个复杂 transformer。那会再次让系统变得难以诊断。

首先围绕 event program 的少量参数做局部 system identification：

$$
u_k \pm \epsilon_i
$$

使用 paired random streams 生成反事实响应，从而估计局部 Jacobian：

$$
r(u_k+\Delta u)
\approx
r_k+A_k\Delta u.
$$

然后求解：

$$
\begin{aligned}
\min_{\Delta u}\quad
&
\|
W(r^\star-r_k-A_k\Delta u)
\|^2
+
\lambda\|\Delta u\|^2
+
\eta J_{\mathrm{deform}}
\\
\text{s.t.}\quad
&
\|\Delta u\|_\infty\leq\delta,
\\
&
u_k+\Delta u
\in\mathcal U_{\mathrm{packet}}.
\end{aligned}
$$

控制变量只需要是：

* residual strength；
* event center shift；
* duration scale；
* root lateral offset；
* torso-yaw scale；
* foot-height scale；
* lead and recovery distance。

这个 optimizer 有三重用途：

1. 立即成为比 scalar secant 更强的 baseline；
2. 产生成功修复轨迹；
3. 为 RepairNet 提供监督数据。

最终比较：

* current scalar repair；
* local response optimizer；
* learned RepairNet。

如果 learned RepairNet 用一次前向传播达到 optimizer 两三次采样的效果，你们的 learning contribution 就成立了。

---

# 五、把 curriculum learning 用在真正适合的位置

你们非常适合在这里加入一个 **repairability-aware curriculum**，但不要把它做成另一个庞大子系统。

训练数据不要按"障碍物数量越多越难"简单排序，而应当按 response complexity 排序：

### Level 0：单调、单轴响应

* overhead duck；
* 只有 amplitude 和 timing；
* 当前 secant repair 可以当 teacher。

### Level 1：phase-sensitive adaptation

* step-over；
* 需要选择 swing foot；
* 需要对齐 obstacle position；
* 需要保持 stance contact。

### Level 2：多身体区域耦合

* lateral squeeze；
* torso yaw + arm tuck + route offset；
* 单独一个 scalar 不够。

### Level 3：多事件组合

* beam 后紧接 floor obstacle；
* step-over 后立刻 turn；
* 前一动作的恢复影响下一动作。

### Level 4：离散策略与连续参数共同选择

* go-around vs duck；
* left-step vs right-step；
* turn-left vs turn-right。

### Level 5：cross-prior transfer

* ARDY 产生的 repair data；
* RepairNet zero-shot 用于 Kimodo；
* 再用少量 Kimodo outcomes 做 adaptation。

采样重点应放在：

> one-shot 失败，但 optimizer 在 1–2 次修复内可以成功的场景。

这类场景正好位于 repair policy 的学习区间。完全简单的场景没有学习信号，完全不可解的场景只有 refusal 信号。

可以定义：

$$
D_{\mathrm{repair}}
=
w_1 d_{\mathrm{clearance}}
+
w_2 d_{\mathrm{phase}}
+
w_3 d_{\mathrm{contact}}
+
w_4 d_{\mathrm{execution}}
+
w_5 d_{\mathrm{mode}}.
$$

然后维持三类池：

* already solved；
* repairable frontier；
* currently unsolved。

训练主要分配给 repairable frontier，并随着策略变强逐步扩展。这会把你过去 curriculum-learning 的专长真正带入 Scene2Motion，而不是只作为场景随机化。

---

# 六、四个决定论文是否变强的实验

## 实验一：coherent residual 是否真的解锁新能力

先不要做大规模 text battery。直接用三个任务：

* overhead duck；
* floor step-over；
* lateral squeeze / turn-and-tuck。

每个任务比较：

| 方法                              | 约束表示                              |
| ------------------------------- | --------------------------------- |
| Current                         | 当前孤立 synthetic constraints        |
| Absolute packet                 | 绝对 donor motion packet            |
| Residual packet                 | phase-matched adaptation residual |
| Residual + one response repair  | residual packet + optimizer       |
| Residual + two response repairs | residual packet + optimizer       |

step-over primary endpoint 必须是：

* obstacle-centered whole-body box clearance；
* crossing position error；
* correct swing foot；
* stance-foot contact；
* SONIC execution success。

不是 foot peak。

这个实验若成功，会直接回答当前最大问题：

> 是 prior 没有能力，还是我们一直在用错误的动作表示访问它？

---

## 实验二：feedback 是否真的优于 equal-budget sampling

固定 generator call budget：

$$
B\in\{1,2,3\}.
$$

比较：

* one-shot；
* independent best-of-\(B\)；
* current monotone repair；
* local response optimizer；
* RepairNet。

核心比较是：

$$
\text{RepairNet-3}
\quad\text{vs}\quad
\text{Best-of-3}.
$$

Texedo 已经使 verifier-based selection 成为必须认真对待的 baseline，因此你们必须证明 observed failure 被用于 **改变后续 program**，并且这比仅仅多采样更有效。

指标不能只有 success：

* executed traversal success；
* minimum executed clearance；
* integrated deformation；
* path length；
* generator calls；
* wall-clock latency；
* contact violations；
* refusal rate。

---

## 实验三：ARDY → Kimodo 的 zero-shot response transfer

Kimodo 不应该只被用来复制"6× overcount"。

最有科研价值的问题是：

> 在 ARDY outcomes 上训练的 RepairNet，是否能根据 Kimodo 的实际 response，zero-shot 修正 Kimodo？

这正是 response-conditioned formulation 的优势。网络不需要精确模拟 Kimodo 内部，只需要看到：

* 当前 program；
* Kimodo 实际生成结果；
* 几何 deficit；
* timing/contact error。

比较：

* scene-only proposer cross-prior transfer；
* open-loop forward model transfer；
* response-conditioned RepairNet transfer。

如果 RepairNet transfer 保留明显收益，就能支持一个很强的主张：

> **Feedback-conditioned repair is more generator-portable than open-loop command prediction.**

但表述仍应限制为"两种共享 G1 embodiment 的 released priors"，不要写 arbitrary prior。

---

## 实验四：真正的 route–body joint planning

如果论文标题里保留 "planner"，就需要让 body adaptation 反过来影响 route selection。

对路径 \(\pi\) 定义：

$$
C(\pi)
=
L(\pi)
+
\lambda
\hat J_{\mathrm{adapt}}(\pi)
+
\eta
\hat p_{\mathrm{exec-fail}}(\pi)
+
\kappa
\hat N_{\mathrm{calls}}(\pi).
$$

比较：

* shortest path；
* current mode-cost A*；
* RAMP-aware route cost；
* oracle over generated and verified routes。

场景必须出现真实 trade-off：

* 短路需要深蹲；
* 长路可以直立；
* 左路需要侧身；
* 右路需要 step-over；
* 多障碍路线需要动作组合。

如果只是给固定 route 调整 pelvis height，最好叫：

> local whole-body adaptation planner

而不是 general navigation planner。

---

# 七、当前 draft 里最危险的具体 claim

这些不是措辞洁癖，而是会直接影响审稿人信任。

## 1. "43-dimensional interface collapses to one axis"

当前证据更准确的说法是：

> Within the tested request families, sampler, robot embodiment, and task metrics, only the absolute-height axis was consistently spatially addressable and executable.

否则 reviewer 会说你没有穷举 43 维组合，也没有证明 Kimodo、其他 prompt family、其他 constraint packets 都坍缩。

## 2. "The teacher is the ceiling"

这句话太绝对。

你们证明的是：

> The particular fitted QP response model remained optimistic under this beam-count and scene shift; its distilled TCN could not recover information absent from that teacher.

不能推导成：

* 更多数据永远无法解决；
* 更好的 teacher 无法解决；
* response-conditioned learned model也无法解决。

恰恰相反，新方法就是要学习当前 teacher 不包含的 measured response。

## 3. "Exact collision geometry"

你们使用的是 shipped MuJoCo collision primitives 加上 4 cm dilation，不是视觉 mesh 的 exact geometry。

改成：

> collision checking against the robot's simulation collision model, with a separately measured geometric coverage margin.

## 4. "Repairs every proposal failure"

在 8-seed v2 结果完全出来之前，不能出现在 abstract。

即使最终是 100%，也应写：

> repairs \(X/Y\) one-shot failures under a fixed call budget

而不是"every proposal failure"，因为后者容易被解释成系统级保证。

## 5. "~10⁵ verified traversals per GPU-day"

需要把 throughput 拆成至少四级：

1. raw generated candidates/day；
2. kinematically scored candidates/day；
3. accepted scene records/day；
4. SONIC-executed records/day。

按照你们自己的运行信息，完整 scene 需要 1–3 s，SONIC 还有独立启动和 rollout 开销，所以 \(10^5\) 只能合理对应某种 batched kinematic-candidate throughput，而不能含糊地写成完整 physics-verified traversal throughput。

最可信的写法是报告实测：

* elapsed wall-clock；
* number generated；
* number accepted；
* number physically executed；
* GPU utilization；
* batching configuration。

## 6. "Defect catalogue is a contribution in itself"

它是很好的 artifact，也会增加信任，但不应成为主 contribution。

顶会 reviewer 很可能认为：

> 修复自己实验中的 bug 是必要科研过程，而不是算法创新。

保留一个 concise provenance paragraph，把完整 catalogue 移到 appendix/project page。它的作用是支撑结论，不是替代结论。

## 7. 数据生成器 claim 当前缺少 downstream utility

CLAW 已经明确提出 scalable、physically grounded、language-annotated G1 motion data generation pipeline，并用 composable primitives 与 SONIC 生成轨迹。GenTrack 又进一步通过 generator–tracker online co-training 改善 robot-executable motion。

所以，仅仅说：

> 我们也能生成很多带标签动作。

并不够。

你们必须证明 Scene2Motion 数据独有的东西带来下游收益，例如：

* 使用 raw ARDY data 训练 RepairNet；
* 使用 only-success verified data；
* 使用 full repair trajectories + failures + deficits；
* 比较 held-out traversal success 和 sample efficiency。

最自然的 downstream experiment 其实就是 RepairNet 本身：

> 数据流水线自动产生 \((scene, program, response, repair, outcome)\) transition；随着数据量增加，RepairNet 的成功率提高、generator calls 下降。

这会让"数据生成"从 speculative artifact 变成方法闭环的一部分。

---

# 八、统计层面还需要一个重要修正

Table 2 的 8 seeds × 36 scenes 不能简单把 288 行都当成独立样本做 McNemar。

seed 是嵌套在 scene 内的重复测量。主要推断单位应当是 scene。

更稳妥的方式是：

* 以 scene-level mean success 为主要统计单位；
* cluster bootstrap over scenes；
* 或 mixed-effects logistic regression，scene 作为 random effect；
* seeds 用于估计 within-scene stochasticity；
* calibration split 必须按 scene/topology，而不是按 clip。

McNemar 可以作为辅助 paired test，但不要让它独自支撑主结论。

同样，conformal calibration 需要 scene-level independent calibration/test split。若样本不够，不要使用"certificate"这种容易让人联想到强形式保证的词，可以写：

> execution-calibrated acceptance rule

并报告 held-out risk–coverage。

---

# 九、论文结构应当怎样重写

## 建议标题

首选：

# **Scene2Motion: Response-Adaptive Motion Programs for Frozen Humanoid Priors**

备选：

# **Scene2Motion: Event-Aligned Adaptation of Frozen Humanoid Motion Priors for 3D Traversal**

原来的：

> A Frozen Motion Prior Is Not a Planner

可以留作 Introduction 里的 punch line，但不要做标题。它太像负面评测，也会引导审稿人寻找 strawman。

## 新的三项贡献

不要五项，压成三项：

### 1. Prior-compatible scene motion programs

一种 event-aligned、phase-aware 的 scene adaptation representation，通过 coherent residual packets 将 duck、step-over 和 squeeze 等动作放到正确的路径位置和 gait phase。

### 2. Response-adaptive repair

一种基于实际生成 clearance/contact/timing response 的局部 optimizer 与 learned RepairNet，在固定生成预算下主动修改 motion program，而不是只做 best-of-\(N\) selection。

### 3. Execution-aware cross-prior planning

利用 SONIC outcomes 学习 executed-clearance model，并在 ARDY 和 Kimodo、程序化场景和 MTC holdouts 上验证 route–body planning、cross-prior transfer 和物理执行。

audit 结果放在 motivating analysis，不再作为第一贡献。

## 新的章节结构

### 1. Introduction

讲 opportunity、hard problem、核心假设与三项贡献。

### 2. Related Work

分成：

* controllable motion priors；
* scene-aware motion planning；
* test-time guidance and verifier selection；
* physics alignment and tracking；
* motion primitives and scene-aligned data。

### 3. Problem Formulation

明确定义 composite stochastic actuator：

$$
\mathcal S,u
\rightarrow
G
\rightarrow
x_{\mathrm{ref}}
\rightarrow
T
\rightarrow
x_{\mathrm{exec}}.
$$

### 4. Scene2Motion-RAMP

* event representation；
* residual packet construction；
* response measurement；
* local optimizer；
* RepairNet；
* execution-aware route cost。

### 5. Experimental Setup

* task families；
* priors；
* trackers；
* splits；
* baselines；
* metrics；
* statistics。

### 6. Main Results

* multi-behavior performance；
* feedback vs sampling；
* cross-prior；
* execution；
* route–body planning。

### 7. Analysis

这里再放：

* naive capability count；
* interface mechanism；
* text does not place behavior；
* failure modes。

### 8. Limitations and Conclusion

artifact、defect catalogue、full ledgers 进入 appendix/project page。

---

# 十、建议的新摘要草稿

> **Large humanoid motion priors provide diverse and natural whole-body behaviors, but their conditioning interfaces do not directly expose a scene-grounded action space: requested adaptations may be distorted, temporally misplaced, or lost during physical tracking. We introduce Scene2Motion, a response-adaptive planner that converts frozen motion priors into scene-conditioned traversal modules without updating their weights. Given a route through 3D clutter, Scene2Motion decomposes the scene into path-aligned interaction events and transports coherent motion-adaptation residuals—such as ducking, stepping over, and lateral squeezing—onto the gait phase of a nominal motion generated by the prior. It then measures whole-body clearance, contact, and event-timing errors and updates the structured motion program using a locally identified response model or an amortized repair policy. A tracker-conditioned outcome model predicts conservative executed clearance and enables execution-aware route selection and receding-horizon replanning. Unlike verifier-only test-time selection, Scene2Motion uses observed failures to modify subsequent generator commands. We evaluate the method with frozen ARDY-G1 and Kimodo-G1 priors, the SONIC whole-body tracker, and held-out procedural and scene-aware traversal environments. Scene2Motion improves executed traversal success by [X], outperforms equal-budget independent resampling by [Y], and transfers response-conditioned repair across priors with [Z] retained improvement. Our analysis further shows that semantic behavior elicitation alone does not imply spatially placed or physically executable traversal, motivating feedback in the realized motion-response space.**

---

# 十一、当前 draft 哪些内容保留，哪些降级

| 当前内容                            | 新位置                                         |
| ------------------------------- | ------------------------------------------- |
| Paired-seed/q99/stability audit | Experimental protocol + analysis            |
| 6× counting result              | 一个 motivating figure，不再是 headline           |
| Scalar duck repair              | 强 baseline / Level-0 controller             |
| TCN proposer                    | open-loop baseline                          |
| §6 interface mechanism          | 设计 residual packet 的依据                      |
| Text step-over failure          | Fig. 1 motivation case                      |
| SONIC achieved-state export     | execution-model data source                 |
| Defect catalogue                | Appendix / artifact                         |
| Provenance frozen ledgers       | Reproducibility paragraph                   |
| §9 generic dataset generator    | 改成 RepairNet data engine，或删除主贡献身份           |
| Refusal                         | 系统输出，不作为独立核心贡献                              |
| Kimodo replication              | 升级为 cross-prior transfer                    |
| MTC                             | 升级为训练 residuals 和 held-out evaluation 的核心资源 |

---

# 十二、现在最值得运行的研究顺序

不要继续优先润色这篇 audit-shaped draft。把它 freeze 成 **baseline technical report**，保留所有数字和 provenance。

下一步研究应按下面的依赖关系推进：

### 第一件事：residual-packet pilot

在 duck、step-over、squeeze 三个行为上比较 synthetic constraint、absolute packet、phase-aligned residual packet。

这是最高信息量的实验，因为它决定：

* 当前瓶颈是 request representation；
* 还是 prior 内部根本没有可寻址能力。

### 第二件事：local response optimizer

把当前 scalar secant 扩展到少量 event parameters，验证反馈能否在多轴任务上超过 best-of-\(N\)。

### 第三件事：RepairNet

用 optimizer 轨迹训练 response-conditioned policy，并用 repairability-aware curriculum 逐步加入 phase、multi-body 和 route-choice complexity。

### 第四件事：SONIC execution model

不只是 gate，而是让 predicted executed outcome 进入 program 和 route optimization。

### 第五件事：ARDY → Kimodo transfer

检验 response-conditioned learning 是否比 open-loop forward prediction 更可迁移。

如果 residual packet 仍然无法可靠表达 step-over/squeeze，不要退回审计论文。那会告诉我们公共 conditioning interface 是瓶颈，下一步应转向：

$$
x_{\mathrm{adapt}}
=
x_{\mathrm{nominal}}
\oplus
A_\psi(\mathcal S,x_{\mathrm{nominal}})
$$

这样的 output-space residual adapter，或者优化 per-window latent。仍然是在设计更强的方法，而不是把失败本身当终点。

---

# 最终判断

这份 draft 最可贵的地方，是它已经把所有"我们为什么以为方法有效"的假象拆得很干净。你们没有白走这段路：现在我们知道不能再做 scene → absolute schedule 的 open-loop imitation，也不能再用 isolated coordinates 去要求协调动作。

但它目前的最主要问题也正是这一点：

> **它把诊断写成了论文，把治疗方案写成了 future work。**

真正能把 Scene2Motion 推向强方法论文的中心转变是：

> 从"审计 prior 有哪些可靠通道"，转向"学习如何通过 event-aligned、prior-compatible motion programs 和 realized-response feedback，主动扩展 prior 在场景中的可用能力"。

当前 Table 2、τ(d)、Kimodo audit 仍然值得完成，但它们应当成为新方法的 baseline、regression test 和 evaluation infrastructure。接下来最重要的结果，不再是 6×、10×或者又发现了几个 defect，而是：

> **coherent residual + response adaptation 能否把 placed step-over、lateral squeeze 和组合 traversal 从失败变成稳定、可执行的成功。**

## References

1. ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation — arXiv 2607.08741
2. CLoSD: Closing the Loop between Simulation and Diffusion for Multi-task Character Control — arXiv 2410.03441
3. Texedo: Test Time Scaling for Controller-aware Language-conditioned Humanoid Motion Generation — arXiv 2606.22998
4. BRIC: Bridging Kinematic Plans and Physical Control at Test Time — arXiv 2511.20431
5. MTC: Moving Through Clutter: Scaling Data Collection and Benchmarking for 3D Scene-Aware Humanoid Locomotion via Virtual Reality — arXiv 2603.05993
6. MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives — arXiv 2604.24833
7. CLAW: Composable Language-Annotated Whole-body Motion Generation — arXiv 2604.11251
