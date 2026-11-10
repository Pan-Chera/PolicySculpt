# 方法来源与实践教训

查阅日期2026-11-05。此skill是机器人方法改编，不是原研究逐项复现，不承诺真机收益。

## 一手来源

- [作者文章](https://rikiyat.github.io/blog/weight-space-programming/)：WSP直接读写权重，不同实验的工具/反馈条件不同。
- [汽车实验协议](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md)及[修改代码](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/answer.md)：跨层重写少量单元连接，实际6073项，形成run-up与退出通路。51条语句不等于51参数；其单次无工具条件不是所有WSP实验的要求。
- [数学实验协议](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)及[50-pass最终说明](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/final_answer.md)：允许有限样本前向/激活；使用方向、层隔离、幅度与随机对照，开发与最终测试分开。pass可含大量序列，不等于一个机器人动作。
- [Gemma负结果](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/gemma_followup/README.md)：直接编辑未总能获得可靠改善，不能假定总能补齐缺失能力。
- [Offline RL教程与综述](https://arxiv.org/abs/2005.01643)：已有数据的分布支持挑战；本skill不因此默认引入离线RL训练。

严格复现时固定实际取得的源文件hash与部署dtype，记录可取得commit，不虚构未取得的源码。动态main不是不可变版本。作者解释、公开成绩、本地复算和推断分开。

## 复用边界与补齐方法

原研究是多种任务/反馈条件下的实验族，不是统一的参数微调算法。本skill把汽车实验的跨层路由和数学实验的方向/组件编辑整理成[机制干预流程](intervention-workflow.md)，并提供[可选工具接入](open-source-tools.md)。时间包络解析属于我们针对机器人结构的改编，不是作者给出的通用机器人公式。

作者补丁应用成功、agent自行发现未知修复、我们自身policy的开发效果、独立真机收益分别验收。固定预算、保留错误、科学对照和人执行真机的流程继续适用；不为照搬某个实验取消离线工具，也不把零工具限制强加给所有WSP任务。

这些是证据类别，不是每个任务都要依次通关的流程。探索/筛查只运行当前问题需要的模块；旧候选的完整独立测试不能成为一切新机制分析的前置等待。按[研究效率](research-efficiency.md)记录成本、复盘与优先级转向，不能把主动提前结束的固定终点实验改称已完成或无效证明。

pyvene、LEACE、weightspace、APRNN、EMMET和LeRobot的一手链接集中在工具流程。集成记录需区分`source_reviewed`、`installed`、`synthetic_tested`、`policy_adapter_tested`、`human_robot_confirmed`，未完成的层级不得自动填通过。当前附带NumPy工具的合成验收不证明上述第三方包均已安装或在机器人policy上跑通。

## 机器人仿真实践经验（不是真机验收）

产物位置、SHA256和核验范围见 [经验产物索引](experience-evidence.json)。优先查服务器上可核验的归档。L1–L7所引旧报告曾核对内容及哈希，2026-11-09复查时六份报告的原本机路径均不可用，目前只作为有出处的历史经验。

**Goblet 接方入位与时序**（索引 `goblet_receiver_entry_timing`）。从冻结原策略用无梯度解析编辑重建同一历史候选，共改六个标量，将接方入位偏移调整−1.35 mm，并将五个学习阶段的时长缩放至72%。固定观测序列用于核对计算，定向闭环筛查为2/6→6/6，独立新初态配对确认为18/28→25/28，有8例修复、1例退化。预登记的1→0方向为14/22→21/22，单侧精确McNemar p=0.01953。0→1方向为4/6→4/6。方向等权净差约+15.9个百分点，保守95%区间为−36.0至+60.9个百分点。较早两批同候选均未过各自门槛。结论限于该冻结候选、任务及所测仿真初态分布。

- **L1，报告核对**：三scalar改动准确平移内部目标，实际物体落点却反向；解码正确不等于物理因果正确。
- **L2，报告核对**：中段脉冲在错误出现前已退出；先查包络时窗，扩展同类参数块也可能无法提供尾段支撑。
- **L3，报告核对**：尾段改动在同case多次reset中修复失败，正负都有效；支持局部效果，不支持唯一符号/Jacobian解释。
- **L4，报告核对**：已暴露开发8例4/8→5/8，独立双向12初态两版均10/12；开发收益不能当总体提高，小样本零差也不证明无效。
- **L5，报告核对**：u=1多对一的最早行匹配导致保护门停；应分离同状态直接效应与跨闭环差，保留旧停判并预注册修正。
- **L6a，报告核对**：准备计划指向旧base，probe却来自正确base，smoke在模型前向前拒绝；模型指针与清单来源需一致。**L6b，经验性提醒**：初始化/碰撞、执行限制或评价也可能造成失败，未在此索引核对单一对应实验；修实现不能自动算神经能力收益。
- **L7，报告核对**：审计也会耗forward；被动解析不算，整段actor重放要算。不应每次元数据检查都重算全部历史。

不把特定毫弧度、stage号、CPU号、样本数或预算当通用硬门。选择依据是当前policy/硬件/任务与用户授权；skill携带方法，不携带过去实验的权限。
