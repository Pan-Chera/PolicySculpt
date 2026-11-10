# 适配器验收、编辑完整性与随机性

按当前分析通路和结论选择范围；A0–A3是已有证据的标签，不是四个必须依次跑满的阶段。共享、缓存与随机检查只覆盖实际相关依赖，结果未失效即可复用。

## 适配器证据状态

验收记录绑定base、runtime/编码/归一化、schema、dtype/device及分析范围；不要沿用另一版本的“通过”。

| 状态 | 最小证据 | 允许的结论 | 尚不能据此声称 |
|---|---|---|---|
| A0 结构/参数可读 | 权重/源码/计算路径及已知缺项 | 提假说、解析预测与适配性建议 | 已观测候选动作改善、原policy已重建 |
| A1 固定输入可比 | 相关输入、必要pre-state/RNG语义与base输出在预定容差内复现，保存误差/范围 | 报已测固定条件下的计算差分 | 自由状态递推、全轨迹或物理收益 |
| A2 所需状态/随机过程可检验 | 本问题需要的reset/hidden/history/chunk/阶段递推；随机协议及证据，或明确不适用 | 在范围内报告序列计算/随机分布比较 | 候选动作实际导致了日志中的物理后果 |
| A3 可交付候选计算已检查 | 去hook保存重载、适用状态检查、编辑完整性和本阶段能力保留结果 | 候选文件实现了所测计算效果 | 未测状态、独立成功率、真机安全/部署 |

无状态/无随机模块对应项写“不适用”，不用增加伪测试。A1只复现某固定输入时，不能扩称覆盖全部policy；随机输出只能分布性复现时，明确分布口径，不能写逐值重建。state足够做局部比较但未检查递推时留在对应范围，不为拿A2/A3标签无限补数据。

保存：验收范围ID、身份、实际测试数据/窗口、容差和最差误差、状态及随机语义、证据文件引用、缺项。候选/运行代码或相关恢复逻辑改变才重做受影响检查。物理案例与独立性能结论需要另有真实执行记录；适配器等级本身不授予这些结论。

反过来，人提供的真实闭环结果可以支持相应案例/完成设计的描述，而不自动提高适配器等级；不要为描述已发生的结果先补跑A0–A3全套。纯权重归因、参数机制和泛化等更强结论仍分别需要对应证据。

## 编辑完整性的最小核验

1. **参数对象与共享存储**：目标键可能是同一个Parameter、同一module的重复引用，或不同视图共享storage。列出目标及别名组、shape/stride/offset、实际消费模块。相同storage不代表访问元素必重叠；不把data pointer相同直接当同一计算角色。
2. **有效权重与包装**：核是否存在weight/spectral normalization、parametrization、量化/打包、低秩组合、共享embedding/head或编译副本。说明编辑的是original还是有效权重、何时重算；改property临时值可能下次forward就失效。
3. **缓存与派生状态**：列出会受编辑影响的KV、RNN hidden、滤波/积分、chunk、预计算embedding/投影、parametrization/编译缓存。按依赖重建/失效；不能复用由旧候选产生的状态，再称新候选自由递推。单步固定base状态干预仍可做，但注明conditioning和可达性限制。
4. **保存重载**：检查数值、共享/别名拓扑、包装设置及行为。文件字节hash变化不必然数值变化；数值相同也不证明绑定关系保留。独立文件格式或逐键clone可能丢共享关系，按目标架构加载后重新核验。重载不应留下旧runtime缓存。

PyTorch接入时先核实际版本：[Module API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)的`named_parameters(remove_duplicate=False)`、`named_buffers(remove_duplicate=False)`可暴露重复命名；默认去重会漏掉部分名称。`state_dict()`不是独立快照，冻结base须有真实隔离副本。存储检查可结合`untyped_storage()`、`storage_offset()`、`stride()`与对象关系，跨进程只比较命名拓扑，不比较瞬时地址。[序列化语义](https://docs.pytorch.org/docs/2.15/notes/serialization.html)

[parametrize源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrize.py)提供`is_parametrized`及缓存上下文；核验实际effective weight与original的关系，不盲目移除包装。上面是API接线提示，不是已在本机真实policy运行过的审计器。

报告物理唯一参数变化数、state_dict受影响名称及共享消费者，不仅给一个“改了N个键”。若本就无共享/包装/相关缓存，记录源码或对象检查依据即可，不制造额外实验。改变共享关系或添加bias/包装属于额外变更，需列出，不再声称只改预定局部参数。

<a id="rng-protocol"></a>
## 随机性协议：同seed不自动等于同扰动

先区分两个问题：①给两版同一随机输入，编辑改变什么；②换随机输入后，结论是否稳定。第一个是控制条件下的计算对照，第二个是跨扰动重复性，不是物理重复性。

- 枚举实际随机源：Python/NumPy/Torch CPU/每设备generator、policy采样器、噪声/扩散步骤、worker、dropout及外部传感扰动；记录版本/device、调用位置与语义。只核当前通路相关来源，不枚举整个系统的无关模块。
- 支持显式随机输入时，记录其内容hash、shape/dtype及语义ID（episode、tick、采样步骤/组件），两版复用同一份。控制动作分布采样时可共用基础uniform/Gaussian而非强制相同最终动作；用候选分布生成的动作仍可不同。
- 不可注入时，分别恢复独立generator状态，并核两版对应调用的数量、顺序和语义。候选分支改变抽样轨迹时，随后相同seed的样本不能再叫精确配对；显式输入shape/语义不兼容时也不能硬塞旧噪声。
- 外部扰动无法复现或非确定性kernel无法控制时，采用预定有限重复采样、独立/分层描述及不确定性，注明未精确配对。不能仅凭seed或无差异样本宣称逐位等价。
- 先在一小段检查配对是否成立，再按问题选择少量独立噪声检查稳定性；不将二者机械扩成大样本。重复随机前向均计数，缓存键包括噪声/流状态和模式。

PyTorch官方也说明跨版本/平台、CPU/GPU即使同seed也不保证相同结果：[Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)。这些限制不妨碍有边界的分布性比较，只限制可报告的精确配对结论。

## 可检查记录与工具边界

`scripts/check_experiment.py`读取计划及可选结果记录，按所选模式检查最小字段、候选base、开发/确认分组、预算/停止条件、确认候选hash与声明层级。它不加载模型，不推断别名/缓存是否实际处理正确，也不证明adapter已重建或声明真实。记录中列出证据文件，按需核hash；共享/包装实际审计仍由适配器完成。

```bash
python3 scripts/check_experiment.py --plan plan.json
python3 scripts/check_experiment.py --plan plan.json --record record.json --check-artifacts
python3 scripts/check_experiment.py --collection-plan collection-plan.json
```

plan沿用实验模板，填`task_mode`、`stage_purpose`、问题、预算、复盘/停止规则。record从 `assets/experiment-record.template.json` 开始；未用字段可删或明确不适用，不用收齐所有验收等级。`artifacts`的相对路径相对record文件，证据refs用artifact ID；远端文件本地不可访问时不加hash核验选项，并如实记录未核验。工具返回`record_consistency_only`不等于通过科学/安全验收，也不触发审批或新增测试。

record的`data_usage`逐组记录`group_id`、可用的session/case、`used_for`与`exposed_before_freeze`。开发数据可同时用于construction/selection；confirmation不得与这些用途、diagnosis或冻结前已暴露分组混用。声明种类为`structural_hypothesis`、`fixed_input_effect`、`sampled_output_description`、`sequence_effect`、`saved_candidate_effect`、`closed_loop_description`或`independent_performance`；种类不是自动证明自然语言陈述真实。随机计算效应注明`same_perturbation`或`across_perturbations`，只有未控制的单次样本则作输出描述。

确认记录列冻结候选及其后观察到的候选hash；实际改过候选会报不一致。固定终点提前停止可记录`closed_loop_description`，不能声明原独立确认已完成。预算超出会报不一致，但保留实际费用；另行记录真实修订，不能删费或改写旧计划使其“过门”。采集计划入口只核人执行分工、记录/运动模式、有限上限和停止/自检说明，不加载SDK，不验证其动作或安全。
