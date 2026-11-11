# 独立具身策略诊断工具

用于机器人策略轨迹诊断、配对仿真和动作块审计。通过独立命令读取任务的原生轨迹、模型和评估器。权重编辑由上游完成，工具接受任意来源的冻结候选。核心命令只依赖 Python 3.10+ 标准库。原生仿真和视觉推理沿用对应任务的运行环境。

## 轨迹诊断

```bash
python3 embodied.py diagnose BASELINE.jsonl.gz CANDIDATE.jsonl.gz \
  --format canonical --threshold object_xyz_m=0.001
```

选用 `--format goblet` 或 `--format cube` 可直接读取已有原生日志。其他任务可提供 JSONL，每行包含 `tick`、`phase`、`signals` 和 `events`。例：

```json
{"tick":240,"phase":"handoff","signals":{"object_xyz_m":[0.18,0.02,0.29]},"events":{"contact":false}}
```

工具按阶段、阶段出现次序和相对 tick 对齐，报告未对齐范围、首个达到阈值的信号差、分量前后值，以及两边事件各自的首次出现。阈值使用字段的原生单位。数据可含独立的策略输出、投影指令、实际应用指令和传感器读回；按日志证据给字段命名。Goblet 的 `projected_joint_reference` 是投影参考，`measured_joint_rad` 是实测状态。缺失字段减少该信号的覆盖，明确记录的 `false` 表示观察到事件未发生。闭环轨迹对齐提供诊断线索，权重归因需要同状态前向或其他对照证据。

## 统一配对评估

复算已有任务，用同一入口中断和继续：

```bash
python3 embodied.py recorded-manifest goblet /path/to/rsi-xarm7-g2 > goblet.json
python3 embodied.py evaluate goblet.json results/goblet --limit-arms 7
python3 embodied.py evaluate goblet.json results/goblet

python3 embodied.py recorded-manifest cube /path/to/rsi-xarm7-g2 > cube.json
python3 embodied.py evaluate cube.json results/cube
```

`recorded` 模式读取既有回执。Goblet 交叉核对模型哈希、seed、方向和原生初态；Cube2 沿用冻结的原求解序列、模型、原生回执和验收规则。

从同一入口实际运行一个冻结 Cube2 样本：

```bash
python3 embodied.py cube-simulation-manifest /path/to/rsi-xarm7-g2 6 \
  --runtime /path/to/rsi-xarm7-g2/.environments/black-bowl/bin/python > cube-live.json
python3 embodied.py evaluate cube-live.json results/cube-live --limit-arms 1
python3 embodied.py evaluate cube-live.json results/cube-live
```

第二种方式会运行 MuJoCo。接入其他仿真时使用 `mode: simulation`，在清单中提供两版 `models`、`evaluator` 文件引用、`command`（`argv`、`cwd`、`timeout_s` 和可选 `environment`），以及包含 `id` 和原生 `initial` 的样本。文件引用形如 `{"path":"/absolute/file","sha256":"…"}`。把解释器、适配器、原生评估代码、配置和相关资产纳入 evaluator 绑定。

命令从 stdin 接收一个 JSON 请求，包含 `case_id`、`role`、模型文件引用、原生初态及其哈希。stdout 只返回一个 JSON，包含请求哈希、实际加载的模型文件哈希、传给原生环境的初态哈希和 `result`。初态哈希绑定原生重置输入，若需要证明重置后状态，另在结果附原生读回。有效结果用 `status: valid` 和布尔 `success`；技术无效与外部安全停止分别用 `technical_invalid`、`safety_stop`。模拟器的任务内碰撞判据可以按原评分规则计入失败，同时单独记录相应标志。

每个样本的每一侧回执独立提交，续跑跳过已验证的回执。两侧均有效才计入配对成功、修复和退化。模型、评估器、初态或回执变动会拒绝沿用原目录。输出 `summary.json` 及绑定后的回执。该命令仅面向仿真或记录导入，真机执行由项目的原生流程负责。

## 动作块和视觉 ACT

```bash
python3 embodied.py audit-chunks TRACE.jsonl

/path/to/native/python embodied.py visual-act JOB_DIR JOB_DIR/demos/validation_00_0.npz \
  --runtime /path/to/native/python

/path/to/sim/python embodied.py visual-act-sim JOB_DIR \
  --actor-runtime /path/to/native/python --groups 3
```

动作块日志可以分别记录生成、送出和实际应用的动作。字段为 `generated: {id, actions}`、`sent: {id, index, action}` 和 `applied: {id, index, action}`，同一 tick 可出现多个字段。审计报告重规划间隔、各块的可见前缀、陈旧块、跳步和动作值变化。控制器的限幅与插值可以产生真实的动作差；解释差值时参照原生控制规则。

`visual-act` 使用已保存的 Cube2 ACT 模型、双相机图像和关节历史，离线重建完整动作块，再和原生推理子进程返回的前缀逐值对齐。它验证固定观测上的预测和前缀传输。

`visual-act-sim` 使用原生仿真与控制器，额外记录每个物理子步的插值目标和实际应用指令。复用任务的 EGL 软件渲染配置和两个原生 Python 环境。默认三次重规划共 24 个物理子步，用于审计动作传递和控制器投影。正式任务评分需按冻结的完整评估方案执行。

## 已验证

| 验收项 | 结果 |
| --- | --- |
| Goblet 历史配对 | 28 对，18 → 25，8 修复、1 退化 |
| Cube2 历史配对 | 30 对，21 → 25，4 修复、0 退化 |
| Cube2 第 06 局原生仿真 | 原策略失败，候选完成。第二次只运行候选侧 |
| Cube2 视觉 ACT 离线 | 623,720 参数，三次均生成 12 步，原生返回的 2 步前缀逐值一致 |
| Cube2 视觉 ACT 原生探针 | 24 个物理子步，三次均生成 12 步、返回并执行 2 步，记录了控制器投影后的应用指令 |
| 测试 | 核对阶段对齐、缺失字段、动作块审计、单侧续跑及冻结身份失配 |

运行本地测试：`python3 -m unittest discover -s . -v`。
