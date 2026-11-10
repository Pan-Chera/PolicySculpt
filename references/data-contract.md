# 日志契约与离线运行

## 最小交接

取得可信checkpoint、冻结推理代码、输入编码/归一化、实际动作ABI与日志来源。先找已有资产再询问缺项，不为通用化强制重训。加载不可信pickle会执行代码，优先安全权重格式/受限加载；非受限加载仅用于已核实可信来源。

`assets/dataset.template.json` 是契约实例，不是通用机器人接口。`template_only=true` 的模板不能准入；填入真实元数据、文件hash与日志后再置false。

| 元数据 | 用途 |
|---|---|
| model/runtime/controller/calibration identity | 分离模型问题与代码、设备、标定变化 |
| observation/action schema、单位、参考系、时钟与频率 | 避免rad/degree、绝对/增量、TCP/grasp或时序混淆 |
| episode/session/physical_case_group | 防重复reset、相邻片段、同采集批次泄漏 |
| log hash、real_robot/simulation/synthetic来源 | 固定证据，不将仿真验收写成真机验收 |
| observation白名单、必需字段、缺失与估计不确定性 | 缺项不默默填真值 |
| proposed/accepted/executed命令及来源、实测读回 | 不能把神经输出当已执行动作 |
| pre-state来源和timing、reset/RNG/动作块 | 恢复循环、随机或分块policy计算上下文 |
| outcome/终止/超时及注释可信度 | 评价属于behavior policy，不能挪给候选 |

位姿可以是传感估计，保留置信度与标定。policy需要物体位姿而日志没有时，拒绝完整前向或声明受限分析，不能用未来真值补齐。真实触觉若原属部署输入可以使用，仿真接触真值不能冒充触觉。同步保存发送、接受、应用和采样时间；驱动确认不证明跟踪准确。

## JSONL行

每行包含episode_id、tick、单调timestamp_s、observation、commands，可另有internal_state与audit。observation仅白名单字段；audit可以有outcome/教师/接触证据但不得传给policy。缺失命令用null并声明原因，不用零向量冒充。

commands的proposed/accepted/executed各为 `{value, provenance}`；传感器读回单列。动作前观测与本周期命令对齐，下一个观测是旧命令后果。过滤后不能把缺tick拼成完整循环轨迹。动作块需记录生成时刻、chunk索引、缓存/聚合与真正发送的子动作，不将每个servo tick算作一次新policy调用。

## 两个分析模式

**logged_state_one_step**：固定 `o_t, h_pre, actual_history, phase/age` 比较两模型。post-state必须按真实源码转换或从前缀恢复，不能直接当pre-state。将base pre-state注入候选是局部计算干预；新权重下该状态可能不可达，需要注明。

**fixed_observation_sequence**：各模型从reset或可证重建状态开始递推内部hidden/phase，物理观测仍固定。若冻结behavior phase日程隔离动作层，写明conditioning，不能称完整policy自由递推。previous-action用实际执行命令还是候选生成命令，必须命名；后者与固定物理观测通常不一致，仅为诊断。

随机policy/diffusion采样/dropout需固定并记录RNG/配置；无法恢复时报告重复前向范围，不将抽样差当权重效应。保持部署dtype、解码和clamp；训练模式和BatchNorm统计变化会污染对照。诊断hook额外运行整层也要计调用。

同seed不自动保证对应扰动相同。按[随机性协议](adapter-and-integrity.md#rng-protocol)选择显式随机输入、核验随机流或有界重复采样，分别说明同扰动效应与跨扰动稳定性。适配器结论按[验收状态](adapter-and-integrity.md)限于实际已重建范围；共享/视图/包装及缓存审计随目标通路检查，不只数state_dict键。

## 实现模型适配器

复用原推理编码/解码，提供三个明确操作：`reset(metadata)`、`snapshot_pre_state()`、`step(observation, state, rng) -> (raw_output, decoded_command, next_state, diagnostics)`。状态包含hidden、滤波/积分、上一动作、阶段与动作块缓存等实际依赖；不只保存RNN hidden。若原runtime是可变对象，先隔离实例并深拷贝状态，不能让base和候选共享缓存。

先在无环境、无驱动进程内重建base，保存逐帧误差与第一个超容差位置，再加入候选：单步模式向两个独立实例注入同一pre-state；序列模式分别reset，并记录各自历史语义。外部观测冻结、内部状态演化、行为日程固定三者写到报告，不能用一个“replay”标签混淆。只导出白名单输入；生成报告可读audit，执行policy的函数不可读audit。白名单检查仅限顶层字段，嵌套结构/shape/dtype/标量范围由适配器按真实schema检查。

输出本轮所需逐帧数值差分、阶段/饱和/退出诊断、episode聚合及最差窗口；保留动作前后语义。晋级候选应有部署精度下的实际张量差分、相关base重建证据、作用窗与适用的成功保留结果。若模块依赖机器人驱动，先提供脱离驱动的纯推理入口；不能为了“离线运行”启动运动服务。此skill附带的是契约校验器，具体模型适配器须按实际源码实现，不宣称已有通用推理引擎。

以上报告范围随本轮阶段选择：内存初筛可只保存代表窗口与差分；晋级候选再扩展适用的保留面板和保存重载。相同base/runtime/输入/状态/dtype/RNG已验证的前向可缓存，按依赖失效重算，不为每个候选重复base全轨迹。缓存与必要前缀规则见[研究效率](research-efficiency.md)。

随附另有[单层仿射折叠工具](open-source-tools.md#affine-helper)，只处理已给定P/c下的局部代数与数值检查。实际adapter仍须支持所需模型重建、状态隔离与适用模式；晋级候选去hook保存重载，涉及状态的候选检查递推/退出，静态网络不重复跑无额外信息的模式。不能拿工具的等价报告代替实际runtime验收。

## 契约检查

```bash
python3 scripts/validate_bundle.py --manifest dataset.json --check-logs --evidence evidence.json
```

脚本只用标准库读取JSON/JSONL/hash，检查身份格式、严格时序、白名单、有限值、命令来源和split冲突，不运行模型或环境。`metadata_valid_only`不是性能门，也不能保证用户填写的provenance真实。

这是完整契约的严格检查，不要求所有受限分析都先拿到通过结果。日志有缺项时说明哪些比较仍成立、哪些无法重建，继续成立的静态/局部分析；不伪造hash、命令或pre-state使校验器过门，也不把受限结果说成完整验收。

同session或同物理case的episode必须同split。已暴露episode补标签后仍暴露。分组导致数据过少时只交开发/回归证据，不为凑holdout偷放宽。未来真机适配需要真实日志核验时钟、实际指令链、标定与感知缺失；合成/仿真工具测试不证明这些条件。
