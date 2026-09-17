# 开发与代码检查

## 修改边界

规范整理应保持算法、阈值、默认参数、输出字段和有效配置值不变。
修改原本报错的分支时，应说明触发条件，并补充针对该分支的验证。
保留第三方源码的版权、许可证和来源说明。

## Python

仓库 Ruff 基线为 Python 3.10、88 字符格式化宽度，以及 `E`、`F`、`W`、`I`
规则；长 URL、日志和文档字符串不强制执行 `E501`。
各模块的运行环境仍以模块自身的依赖说明为准。

- 使用 Ruff 统一格式和导入分组。
- 类型注解应描述实际输入与返回值；默认值为 `None` 时显式标注可选类型。
- 动态注册的模型缓冲区、配置和数据字段应补充准确的类型声明。
- 不通过删除有副作用的调用、导入或异常处理来消除检查错误。
- 同名函数重复定义时先确认最后生效的实现；重复字典键应保留实际值及键顺序。
- `noqa` 仅用于具体的导入路径初始化、兼容导出或注册副作用，并注明原因。
  不批量屏蔽未定义名称或类型错误。
- 保留旧 PyTorch 环境所需的 `torch.cuda.amp` 调用，并仅在调用处注明
  `ty: ignore[deprecated]`；这不屏蔽类型错误。框架回调签名与上游注解不一致时，
  必须为具体的忽略项注明原因。

从仓库根目录执行：

```bash
ruff check .
ruff format --check .
```

类型检查按模块执行，避免两个不同模块中的 `config` 包互相遮蔽。
各目录的 `ty.toml` 声明脚本运行时使用的导入路径，不排除业务源码或内置源码。
使用 `--python` 指向已安装对应模块依赖的环境，例如：

```bash
ty check --project perception --python <PERCEPTION_ENV>
ty check --project action_recognition/rule_based_code --python <RULES_ENV>
ty check --project action_recognition/actionclip --python <ACTIONCLIP_ENV>
ty check --project action_recognition/FROSTER --python <FROSTER_ENV>
```

`<..._ENV>` 是虚拟环境目录的占位符。缺失依赖、平台不兼容和源码错误需要分别
处理。macOS 无法运行的 CUDA 扩展可用官方源码补充静态导入路径，例如：

```bash
ty check --project action_recognition/FROSTER --python "$FROSTER_ENV" \
  --extra-search-path "$DETECTRON2_SOURCE" \
  --extra-search-path "$APEX_SOURCE" \
  --extra-search-path "$PLATFORM_TYPES"
```

这里的路径均为对应源码包的父目录：Apex 使用仍包含 `apex.parallel` 的版本
（本次为 23.05）；平台依赖包括 Decord、TensorRT，以及可选标签图读取器所需的
TensorFlow Object Detection API `object_detection.protos`。缺少类型声明的 TensorRT
编译扩展通过明确的 `Protocol` 接口描述，不执行该后端即可检查调用契约。
仅运行 Ruff 或仅通过某个模块的检查，不等于全仓类型检查通过。

## 测试

感知层现有测试：

```bash
cd perception
python -m pytest tests -q
```

FROSTER 的 CPU 分支回归测试，从仓库根目录执行：

```bash
python -m pytest action_recognition/FROSTER/tests -q
```

测试覆盖注意力 dropout/补零、激活函数、五位置裁剪，以及普通训练分支在开启和
关闭梯度值裁剪时与单步 SGD 的参数更新对比；通过隔离可选 GPU 模块导入执行
真实训练函数体，不代表完整训练任务已运行。

感知层测试覆盖输入预检、几何处理和视频帧对齐，不等于 GPU 推理、模型训练或
真实视频端到端验证。需要权重、视频或特定 GPU 后端的验证应单独记录环境与结果。

## README 与注释

- 使用仓库当前目录名和相对路径；示例中的外部路径使用明确的占位符。
- 标注命令的工作目录、输入、输出及可选依赖。
- 代码块注明语言；目录树和纯文本使用 `text`。
- 注释解释约束、单位、坐标系和特殊处理原因，避免复制已过期的实现说明。
- 已验证结果说明验证范围；不要将历史或局部结果描述为全项目保证。
- 保留许可证中的原始文字；普通文档清理行尾空白并以一个换行符结束。

本次检查环境和验证范围见 [规范整理验证记录](CODE_QUALITY.md)。
