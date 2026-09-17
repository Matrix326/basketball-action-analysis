# 规范整理验证记录

## 范围

统一 Python 格式、导入和类型注解，修正重复定义与重复配置键，整理 README、
路径示例和失效注释。保留最后生效的函数和配置值，不调整模型结构、损失公式、
训练超参数、推理阈值或规则判定参数。

修复的原本报错分支包括：缺失的注意力/激活函数引用、普通训练批次解包、
梯度值裁剪后的范数记录、分布式初始化参数、训练统计属性和可视化调用。
`GroupFCSample` 按确认的处理方式复用现有五位置裁剪函数。

## 检查环境

- Ruff 0.16.8，ty 0.0.81，Python 3.12.14；静态语法目标为 Python 3.10。
- CPU 检查环境：PyTorch 2.14.0、torchvision 0.29.0、NumPy 1.26.4、
  MMEngine 0.10.7、mmcv-lite 2.2.0、pytest 9.1.1。
- 类型检查分别覆盖 perception、rule_based_code、actionclip、FROSTER，
  包含仓库内 MMPose/PySlowFast 源码及测试，无整目录排除。
- 可选平台依赖使用官方包/源码补充导入路径；它们没有作为可执行 CUDA 后端测试。
  Apex 源码为 23.05；PyTorchVideo 源码版本为
  `f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0`。
- TensorFlow 标签图协议使用[官方定义](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/string_int_label_map.proto)
  在临时检查环境生成 Python/pyi 文件，未把生成器版本绑定到项目运行环境。

检查环境独立于项目部署环境，没有替换项目依赖锁定版本。
检查命令见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 验证内容

以下检查均已通过；两组测试分别在各自要求的工作目录运行，共 19 项通过。
FROSTER 测试保留两条旧 AMP API 的运行时弃用提示。

- 全仓 `ruff check .`、`ruff format --check .` 和 `git diff --check`。
- 四个模块分别执行 `ty check`；兼容旧版 Torch 的 AMP 弃用提示在具体调用处注明，
  未禁用未定义名称、导入、参数或返回类型检查。
- 规则模块 wheel 构建成功，并确认包含 `action_rules`、`ball_trajectory` 两个包。
- perception 现有 11 项测试。
- FROSTER 新增 8 项 CPU 回归测试。
- 对比 HEAD 与修改后的规则引擎：5 组模拟轨迹（弹道、缺帧、异常点、持球、运球），
  轨迹处理和动作识别的 JSON 输出完全一致。
- 所有已跟踪 Python 文件按 Python 3.10 语法解析，检查类型专用导入未被
  `cast` 的运行时类型表达式误用。

## 验证边界与环境说明

没有执行需要真实视频、模型权重或 CUDA 的端到端推理、完整训练、多卡通信、
元学习和 TensorRT 执行。上述局部测试不能证明所有分支在所有环境下数值等价。

perception 的 requirements 与 pyproject 在 NumPy、ONNX Runtime 版本上仍有差异；
FROSTER 的历史依赖清单包含互斥的 Pillow 固定版本，且安装说明与清单的 Torch
版本不同。本次保留这些实验依赖值并明确记录，未为消除静态检查而升级部署环境。
