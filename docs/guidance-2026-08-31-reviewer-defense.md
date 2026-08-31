# Guidance received 2026-08-31 — reviewer-defense priorities for the paper draft

Saved verbatim (original below). Fifth guidance document; concerns hardening
`docs/paper-draft-v0.md` against three anticipated reviewer attacks, plus a pre-deadline
punch list. Actioned in the draft and in the Phase 5 schedule (see `response-2026-08-29-phase5.md`
§6 ordering changes noted inline here).

## Working summary (EN)

**Attack 1 — repertoire breadth.** "All this machinery and the usable repertoire is duck plus
an unstable step-over — does that support a general navigation/traversal system?" Defense:
(a) frame explicitly in the Introduction that the contribution is the *mechanism and the
methodology/protocol*, with duck as the archetypal — and hardest — continuous adaptation
instance; (b) foreground §9: even duck-only, a pipeline producing ~10⁵ physically-verified,
refusal-labeled traversals per GPU-day in arbitrarily complex scenes is a major practical
contribution for end-to-end RL / distilled-policy training; (c) land the **Kimodo-G1
cross-model validation as soon as possible** — even one short table/paragraph destroys the
"this is just an ARDY-specific defect study" reading. → *Schedule change: 1D runs immediately
after 1B smoke, before the full 1B pass.*

**Attack 2 — kinematic vs dynamic closure (§7).** The draft concedes the certificate is
kinematic while tracking error reaches 71.9 mm at deep duck. Defense: (a) Fig. 5's τ(d)
risk–coverage must be solid — show that trajectories passing the depth-penalized acceptance
gate achieve **≥95 % executable success under dynamics rollout**; (b) the demo must show
**side-by-side physics**: a naive trajectory drifting into the beam under tracking vs the
τ(d)-repaired trajectory passing cleanly. → *Schedule change: 1B gains an explicit acceptance
target (≥0.95 executed success among accepted) and a paired naive-vs-repaired tracked-video
deliverable.*

**Attack 3 — "pure negative / evaluation paper" misread.** Defense: present §3's repair
operator as a **general black-box generative-repair scheme** with stated guarantees — forward
secant gain (matched to the direction the correction moves), anticipation 3τ·v derived from
the measured first-order lag, and strict monotonicity (the correction can only deepen/extend,
so it cannot trade one clearance violation for another) — not an ad-hoc tuning script.

**Punch list.**
- [ ] Table 2 landed: 8-seed McNemar p-values and the zero-regression conclusion robust.
- [ ] Fig. 3 four-panel with real impact: slight-deficit repair · OOD multi-beam two-round
      repair · topology detour · quantified refusal. Fig. 4 channel signal-to-noise: root
      height (clean) vs position (drowned) vs rotation (very high response).
- [ ] Appendix release manifest: defect catalogue as a table —
      *phenomenon → which false-positive/false-negative it would have produced → the control
      that caught it*.

## Original (verbatim)

二、 审稿人可能攻击的弱点与改进建议（Areas for Improvement）1. 动作多样性过窄的质疑（The Repertoire Breadth Risk）风险点：Reviewer 会质疑："通篇花了大力气，最后实际证明可用的只有 Duck（下蹲），外加一个不稳定的 Step-over，这是否足够支撑一个通用的 Navigation & Traversal 系统？"改进策略：在 Introduction 和 Framing 明确强调：本文关注的是底层机制与系统方法论（Methodology & Protocol），下蹲只是最典型、最难的连续自适应动作实例。突出数据生成流水线（§9）的价值：即使只有 Duck，能在任意复杂场景下每 GPU-天产出 $10^5$ 条带真实物理验证、带拒识标签的数据，对训练端到端 RL 或 Distilled Policy 已经是巨大的实际贡献。尽快补上 §10 中提到的 Kimodo-G1 跨模型跨架构验证，哪怕只放一个简短的 Table/Paragraph，也能彻底击碎"这只是 ARDY 自身特定缺陷"的指责。2. 从 Kinematic 到 Dynamic 的闭环证明（Execution-Aware Gap, §7）风险点：草稿坦白了目前主要还是 Kinematic Certificate，而真实的 SONIC Tracking 误差在深蹲时会放大到 71.9 mm。如果审稿人认为"你们虽然修好了运动学轨迹，但物理执行依然可能摔倒/卡住"，闭环的说服力就会打折。改进策略：Fig. 5 的 $\tau(d)$ 风险-覆盖曲线必须扎实：证明带上深度惩罚 margin 后，通过 Acceptance Gate 的轨迹在 MuJoCo 动力学 Rollout 下的 Executable Success Rate 达到 95%+ 以上。视频 Demo 中务必并排展示：Naive 生成轨迹在物理执行中因跟踪漂移卡撞横梁 vs 经 $\tau(d)$ 修复后的轨迹平稳通过。3. 避免被误判为"纯负面/纯评测文章"风险点：如果 Reviewer 带着找创新点的眼光看，可能会误以为这只是一篇"发现某个模型不行"的 Evaluation Paper，而忽视你们提出的 Closed-loop Repair 算法。改进策略：强化 §3 System 的算法贡献：突出 "前向割线增益 + 基于一阶动力学滞后 $3\tau v$ 的提前量补偿 + 严格单调性保证" 是一套通用的黑盒生成修复理论，而不仅是一个 Ad-hoc 的调参脚本。三、 截稿前的关键交付物清单（Punch List）[ ] Table 2 数据就位：确保 8-seed 运行完成后的 McNemar 检验 $p$ 值和零回归（0 regressions）结论保持稳健。[ ] Fig. 3 / Fig. 4 视觉冲击力：Fig. 3 必须清晰体现四联动：轻微亏损修复、OOD 多横梁两轮修复、拓扑绕行、量化拒识；Fig. 4 直观展示通道信噪比（Root Height 极干净 vs Position 淹没在噪声中 vs Rotation 极高响应）。[ ] Appendix Release Manifest：将 Defect Catalogue 按分类整理成表格，列出"错误现象 $\to$ 诱发假阳性假阴性假设 $\to$ 发现它的对照测试"。
