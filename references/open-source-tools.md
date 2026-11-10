# 可选开源工具与具体接入流程

查阅日期2026-11-05。这里只列与现有policy离线修复直接相关的工具；不因调用skill自动安装、联网调用、训练或连接设备。说明基于下列一手文档/源码；第三方包的真实模型集成需在使用时核验，不把“上游声称支持”写成本机已验收。

## 选择入口

| 当前缺口 | 优先工具 | 交付 |
|---|---|---|
| 张量探索、预算probe、diff与回滚 | weightspace，或现有等价工具 | 固定base副本、probe代码、调用账本、patch |
| 内部替换/消融、RNN/MLP通路定位 | pyvene或简单PyTorch静态hook | 干预位置/状态契约、恢复检查、逐窗结果 |
| 有证据的线性概念方向与保留变换 | LEACE，或显式方向投影 | 开发统计、P/c、覆盖/混杂/保留报告 |
| 固定仿射干预写入线性权重 | 本skill的affine_patch.py | 单层NPZ候选、等价报告，后接真实runtime验收 |
| 明确规格下的区域修复 | APRNN（条件适配时） | 规格、求解与原网络验证，单列constraint_repair |
| 低秩保留/修复目标的理论参考 | EMMET/ROME/MEMIT | 方法适用性与是否含梯度的审计 |
| 人执行采集、episode和多模态存储 | 原有运行器优先，必要时LeRobot | 人工操作包、本地日志和扩展状态记录 |

不要求装齐。先记录所选repo URL、实际commit/包版本、依赖、设备/dtype、许可及是否通过本地测试；版本不可取得时如实注明。隔离分析环境，不升级现场控制环境。依赖安装是准备工作，不运行上游默认机器人示例或训练命令。

常用包名为`weightspace`（按需选择`torch`等extra）、`pyvene`、`concept-erasure`，本地折叠工具只需`numpy`。先按所选版本兼容性生成锁定依赖文件，再在独立venv安装；记录`importlib.metadata.version(...)`及源码commit。已有可用分析环境时复用，不重复创建。此skill未附已安装的第三方包，也不提供未经核验的“最新版组合”。

## weightspace：只暴露离线probe

[项目与API](https://github.com/leofmarciano/weightspace)，[源码README](https://raw.githubusercontent.com/leofmarciano/weightspace/main/README.md)。它提供张量切片/SVD、编辑追踪、checkpoint/rollback及自定义预算probe。它是第三方harness，不是原论文成绩或真机安全系统；README明确代码沙箱不是安全边界。

接入顺序：

1. 在隔离环境按已选版本安装必要extra，记录版本与安装结果；仅为当前后端安装依赖，不机械使用`[all]`。从可信本地checkpoint构造纯推理模型并核base哈希，不用机器人运行器。
2. `fix`会原地编辑传入模型；先保留不可变base和独立编辑副本。确认未覆盖真实部署目录，buffer及cache均隔离。
3. 写固定数据范围的probe：只读开发日志的白名单输入，重建pre-state，调用离线适配器；只返回计算误差、假说相关诊断、成功能力保留与实际计数。probe不得访问驱动、启动环境、拉取任意路径或打开确认集。
4. 在probe内部先检查任务真实帧/序列预算，再执行并计失败开销。上游`forward_budget`是probe成本单位，不能代替实际帧/组件/预热账本；需要时设二级预算。预算结束返回可识别停止，不伪造零消耗结果。
5. 如需外部编辑LLM，保留用户指定模型/推理强度；先核权重切片、日志摘要发往该提供商的授权。无导出授权时使用现有获准的本地/会话工具，不把更换harness当成导出许可。
6. 运行有限编辑，保存transcript、stop reason、diff、patch、失败候选与账本；扫描是否有绕probe的手工前向或拟合。冻结后在全新base副本应用补丁、去hook并保存重载，按实际dtype验收，随后交人执行必要新确认。

以下是**接线示例**，`adapter`、`frame_ledger`、`editor_backend`等须由实际工程提供，不能原样执行或声称已实现：

```python
import copy
import weightspace as wsp

editable = copy.deepcopy(base).eval()

@wsp.probe
def inspect_development():
    # 此适配器只能读取冻结开发输入；reserve/actual在内部对每次调用记账。
    return adapter.inspect_fixed_inputs(editable, development_ids, frame_ledger)

result = wsp.fix(
    editable, hypothesis_text, llm=editor_backend,
    context=verified_architecture_and_units,
    probes=[inspect_development], forward_budget=probe_cap,
)
result.save(stage_output_directory)
```

新接入或相关代码变化时，可用上游`ScriptedLLM`或最小确定性控制验证预算/rollback/输出；已有有效结果可复用，不要求每轮重新跑工具验收。不把上游示例`rollout`作为真机采集入口。

## pyvene：静态诊断，不自动训练干预

[项目](https://github.com/stanfordnlp/pyvene)、[文档](https://stanfordnlp.github.io/pyvene/)、[论文](https://arxiv.org/abs/2403.07809)。静态替换/收集/消融适合当前任务；它也有可训练干预，不能默认启用后者。

1. 从已通过重建的纯推理模型列出真实module和tensor轴。优先在拟定位通路两端插入有限观测点，不一次存下所有中间量。
2. 核对选定版本的`IntervenableConfig`、`RepresentationConfig`、`IntervenableModel`及模型映射。自定义机器人网络需要component→真实module/hook/dimension映射；不能把语言模型的`block_output`名称套进GRU。参阅[配置源码](https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/configuration_intervenable_model.py)与[映射说明所在源码](https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/intervenable_base.py)。映射成本不合理时，使用等价的原生静态hook并如实标记，不能宣称已集成pyvene。
3. 先做NO_OP和收集模式，核输出/状态与base一致。采用静态替换/零化/固定方向，关闭参数梯度并用推理模式；禁止训练旋转子空间或用其优化API代替直接权重编辑。
4. 固定episode/tick/调用序号、pre-state和源/目标激活语义，按[干预流程](intervention-workflow.md)执行双向或组件对照。模块返回tuple、GRU fused kernel或分块缓存时检查实际被修改位置，不能只看到hook注册成功就算生效。
5. `finally`移除全部hook与缓存，验证关闭后恢复；比较权重/buffer hash。把干预定义和结果存成报告。若要成为候选，必须另完成永久权重编译与保存重载验证。

无需为了使用该库加载语言模型、联网获取模型或把整套现场控制器封进其wrapper。

## LEACE：从开发统计构造固定仿射变换

[论文](https://arxiv.org/abs/2306.03819)、[实现](https://github.com/EleutherAI/concept-erasure)、[变换源码](https://github.com/EleutherAI/concept-erasure/blob/main/concept_erasure/leace.py)。它估计协方差并构造仿射概念消除变换；线性不可识别性不是闭环修复保证。

1. 固定开发激活`X[N,d]`及有明确定义的概念标签`Z[N,k]`；分类标签须显式one-hot编码，不能把整数类别当连续次序。记录按episode抽样/加权、阶段/位姿匹配、混杂与样本量相对维度，不从确认集估计统计。
2. 用`LeaceEraser.fit(X,Z)`或分批`LeaceFitter.update`估计；保存超参数、统计样本ID、均值、P及实际rank/截断。它是统计干预，不是纯权重阅读，也不是正确动作标签拟合。
3. 先在离线激活做临时干预，检查状态范数、输出/成功保留及独立开发切片，选择必要的sham/随机方向对照。若概念和任务必要状态纠缠，或维度太高/统计不稳，停止此候选，不把“擦掉失败”当作修复。
4. 当前源码定义列向量等价式为`x′=P x+(I−P)μ`（`eraser.bias`存的是中心μ，不是平移c）。不要直接拿该`bias`当下一层新bias；先核当前版本函数实现，再求c。
5. 仅在实际接线满足[折叠条件](intervention-workflow.md)时折叠进既有线性层；晋级候选去hook、保存重载，并通过相关模式与本阶段保留面板后交人确认。涉及状态递推则检查残留，无记忆网络不机械跑第二模式。否则保留为诊断，不偷加部署模块。

以下为开发激活已准备好时的转换片段；PyTorch/LEACE须在隔离环境实际安装测试：

```python
import torch
from concept_erasure import LeaceEraser

with torch.no_grad():
    eraser = LeaceEraser.fit(X_development, Z_development)
    P = eraser.P
    center = eraser.bias
    c = torch.zeros(P.shape[0], device=P.device, dtype=P.dtype)
    if center is not None:
        c = center - P @ center
    # 导出为float64 NPZ前先确认目标路径授权；bias不是c。
    probe_values = eraser(X_diagnostic)
```

<a id="affine-helper"></a>
## 本地单层折叠工具

需要NumPy，无需torch/外部API/机器人。原生标准库日志校验器仍可单独使用。将原层导出为`layer.npz`（`weight[out,in]`和可选`bias[out]`），变换为`transform.npz`（`P[d,d]`、`offset[d]`），开发集的原层输入为`features.npz`（`features[N,in]`）。只能float16/32/64，不读pickle或完整`.pt`；将BF16/量化参数先转float32做局部试验不能冒充原部署精度验收。

```bash
python3 scripts/affine_patch.py --layer layer.npz --transform transform.npz \
  --features features.npz --side input --atol 0.00001 --rtol 0.00001 \
  --out candidate_layer.npz --report affine_report.json
python3 scripts/test_affine_patch.py
```

容差数值仅为命令例子，实际须在看结果前按部署dtype/累加方式确定；不能失败后放宽。`input`把P/c放当前层输入，`output`放当前层输出。输出仅为单层候选，不会更新完整模型；实际tensor路径、转置、参数资格、保存重载与其余参数未变由适配器核验。

工具在受控float64累加、变换/层边界转回层dtype的参考计算中比较两条路径；不是GPU/量化/混合精度模拟器。`local_affine_check_only`只表示给定特征通过局部数值检查。`no_op_after_cast=true`表示编辑被舍入抹掉，不能据此声称实现修复。bias缺失但需要新增bias、非有限值、轴错位、容差失败会拒绝；失败报告保留，已有文件不覆盖。两次线性参考求值各含N行，计2N层帧等价并保留失败消耗，不能因为没有完整actor前向就一律记零。

可导入`edit_output_writers(weight, direction, columns, alpha)`做选定写入列的定向减法；u须事先显式单位化。该函数不改bias和未选列，也不宣称全局概念消除。

## APRNN与EMMET：有条件的补充路线

**[APRNN](https://github.com/95616ARG/APRNN) / [论文](https://arxiv.org/abs/2304.03496)**：适合能给出明确输入区域和输出规格、且网络激活满足其方法条件的子网络。

流程：核激活/算子支持与求解器依赖 → 从真实约束定义区域/输出规格（不能把失败动作当正确标签）→ 固定成功保留规格 → 在隔离副本有限求解 → 用原runtime核规格及域外回归 → 将求解保证的对象/范围、实际误差和未覆盖处一起交付。GRU/tanh/sigmoid不能因都叫神经网络就直接套；修改成ReLU替身后得到的保证不自动适用于原policy。网络区域保证不覆盖物理闭环。按`constraint_repair`单列，不称原WSP逐项复现。

**[EMMET及统一编辑框架](https://github.com/scalable-model-editing/unified-model-editing) / [论文](https://arxiv.org/abs/2403.14236)**：可借鉴“满足编辑目标，同时保留原映射”的低秩/约束思想。

先核目标激活/输出从何而来、矩阵轴与保留统计。默认实现的[compute_z.py](https://github.com/scalable-model-editing/unified-model-editing/blob/main/emmet/compute_z.py)含Adam和`loss.backward()`；不能因最终矩阵更新是解析式就宣称端到端无梯度。当前纯WSP任务不直接运行其默认编辑器。若用户授权比较，单列方法/依赖/训练或拟合费用及对照；改成明确目标的解析约束方案时也说明是方法改编，不把语言token键值寻址照搬机器人状态。

闭式动作标签回归仍是`supervised_fit`。WIZARD一类元训练生成adapter、权重扩散生成器、world model或critic训练，都不是本skill默认补齐步骤。

<a id="lerobot-collection"></a>
## LeRobot：只交人执行采集入口

[原项目](https://github.com/huggingface/lerobot)、[采集文档](https://huggingface.co/docs/lerobot/il_robots)。已有现场运行器够用时只加日志，不为格式统一替换控制器。核设备支持；自定义双臂接口需接入真实Robot/遥操接口，不能照抄SO系列的端口/校准命令。

1. agent在隔离环境开发配置/扩展记录器及转换器，按[人工采集流程](human-collection.md)用合成或已有日志自检；将是否发送运动指令写到操作单。
2. 人核验设备/标定并执行短段采集。提供给人的实际版本命令须固定本地`--dataset.root`、任务/episode上限，并显式关闭`--dataset.push_to_hub=False`；查该版本源码确认其它上传/遥测/后台路径，不把单个flag当绝对无网保证。agent不代执行运动命令、不使用replay自动复位。
3. 补充原始网络输出/接受/实际应用命令、pre-hidden、上一动作、chunk缓存、模型及runtime hash、时间同步与介入记录。LeRobot标准state/action/video本身不保证足以重建这些内容，缺失时单列限制。
4. 保留原episode标号、时间戳和坏/中断记录；面向WSP的JSONL/manifest转换不重编号掩盖掉帧，不把遥操命令说成policy输出。
5. agent核短段base重建后，人按冻结采集单完成；agent回收开发日志做离线修复。候选确认另交冻结版本，由人执行新回合。不开启自动上传、训练或现场权重替换。
