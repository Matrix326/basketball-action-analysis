# Perception：篮球多视角感知与轨迹

本目录负责球员/篮球检测、人物实例轮廓、COCO-17 姿态、跨视角 ReID、3D 重建、轨迹平滑和可视化。入口、配置、模型和运行时源码集中在本目录，可独立运行。

## 目录与流程

```text
basketball-action-analysis/
├── ljy/                         # 原有动作分析，未修改
└── perception/
    ├── README.md
    ├── docs/                    # 输入输出接口
    ├── config/                  # YAML 配置及加载器
    ├── assets/                  # 相机标定与球场背景
    ├── models/                  # 模型权重，二进制文件由 Git LFS 管理
    ├── basketball_repro/        # RF-DETR 推理运行时
    ├── third_party/             # MMPose 运行时及许可证
    ├── src/
    │   ├── check_inputs.py
    │   ├── run_rfdetr_full_pipeline.py
    │   ├── rfdetr_pose_multiview.py
    │   ├── rfdetr_pipeline/
    │   ├── track/
    │   └── generate_reid_3d_multiview.py
    ├── tests/
    └── output/                  # 运行结果，不纳入 Git
```

同步视频 + 相机标定 → RF-DETR 分割/检测 → RTMPose + ReID + 3D 重建 → 原视角/俯视轨迹视频 → 额外轨迹平滑 JSON → 3D 骨架 MP4/GIF。

## 快速运行

先按下文“模型与环境”准备依赖并拉取模型，再运行以下命令。将 `<REPO_ROOT>` 替换为仓库根目录的绝对路径，将 `<ENV_PREFIX>` 替换为已安装依赖的 Python 虚拟环境或 Conda 环境目录；这些是占位符，不要原样执行。环境名称不作限制。

```bash
cd "<REPO_ROOT>/perception"
export PATH="<ENV_PREFIX>/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/output/tmp"
export MPLCONFIGDIR="$PWD/output/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

python src/check_inputs.py \
  --config config/config.yaml --start-frame 900 --end-frame 1200

CUDA_VISIBLE_DEVICES=0 python src/run_rfdetr_full_pipeline.py \
  --config config/config.yaml \
  --start-frame 900 --end-frame 1200 \
  --output-base output/demo_900_1200
```

这会处理同步帧区间 `[900, 1200)`，即每视角 300 帧，当前数据为 30 FPS、10 秒。使用新的输出目录，避免覆盖已有结果。预检成功输出 `"status": "passed"`；失败退出码为 1。预检检查路径、标定相机名称、视频 FPS、可用帧范围和首帧解码，不代替实际 GPU 推理或标定精度评估。

已使用四路真实视频完成同一区间的检测、轨迹、平滑和 3D 动画全流程验证；13 个 MP4 均通过 300 帧、30 FPS 的完整解码检查。该结果不代表任意数据或环境均已验证。

## 换用自己的数据

复制完整的 `config/config.yaml` 到 `config/local.yaml` 并修改下列内容；所有命令的 `--config` 一起替换为新文件。加载器读取单个 YAML，**不会自动把自定义配置与 `default.yaml` 合并**。

| 配置 | 要求 |
|---|---|
| `project_root` | 配置放在本目录的 `config/` 内时保留 `auto`；放在其他位置时填写本目录绝对路径 |
| `data_root`、`videos` | 输入视频路径；将 `data_root` 设置为 `<DATA_ROOT>` 对应的实际视频数据根目录，并逐项检查 `videos`。仓库配置中的原始本机路径必须按实际环境调整 |
| `camera.intrinsics_path`、`extrinsics_path` | 与视频相匹配的内外参，不能给其他场地直接套用当前标定 |
| `camera.view_to_camera` | 视角名对应标定文件中的相机名，默认 view1/2/3/4 → A1/A2/B3/B4 |
| `camera.frame_offsets` | 原视频帧号 = 同步帧号 + 此视角偏移 |
| `trajectory.fps` | 必须与所有输入视频 FPS 一致，主流程不会自动重采样 |
| `camera.court_world_bounds` | 标定世界坐标中的有效场地范围，单位米 |
| `reid.num_players` | 当前默认 6 人，按实际比赛设置 |

输入应为已同步、已去畸变的视频，至少两个有效标定视角，推荐默认四视角。处理区间加上每个偏移后都必须落在对应视频内。更换视频后先运行预检；不要依赖主入口对缺失视频的过滤或推理阶段对末尾帧的截断来保证完整输出。

## 常用运行方式

只输出检测、姿态、ReID 和 3D 数据：

```bash
CUDA_VISIBLE_DEVICES=0 python src/rfdetr_pose_multiview.py \
  --config config/config.yaml --start-frame 900 --end-frame 1200 \
  --output-dir output/detection_900_1200/poses
```

复用上面完整流程的分析结果，重新生成轨迹及动画：

```bash
python src/run_rfdetr_full_pipeline.py \
  --config config/config.yaml --start-frame 900 --end-frame 1200 \
  --output-base output/demo_900_1200 --skip-analysis
```

复用时必须保留相同输入、标定、视角、帧区间和输出根目录。程序不会自动校验这些是否与旧结果一致。仅检测入口写出到 `--output-dir`；完整入口使用 `--output-base`，两者不是同一参数。

其他参数：

- `--limit N`：从起始帧处理 N 个同步帧，优先于 `--end-frame`。
- `--views view1 view2`：只处理这些视角，至少两个；预检时也传相同参数。
- `--skip-smoothing`：跳过额外的轨迹平滑 JSON 阶段，不关闭检测器时序滤波或轨迹视频自身的平滑。
- `--skip-3d-animation`：不生成 3D 动画。
- 不传帧区间时使用 `trajectory.start_frame` 和 `process_seconds * fps`；不传输出根目录时使用配置的输出路径。

## 模型与环境

`models/` 纳入版本控制，其中模型二进制文件通过 Git LFS 管理，旁侧 JSON 元数据使用普通 Git 管理。外部输入视频和 `output/` 不纳入提交，需要自行准备视频并修改配置。

先安装 [Git LFS](https://git-lfs.com/)，克隆后在仓库中执行：

```bash
cd "<REPO_ROOT>"
git lfs install --local
git lfs pull --include="perception/models/**" --exclude=""
git lfs fsck
cd perception
```

`<REPO_ROOT>` 是克隆后的仓库根目录。Git LFS 未安装或未拉取成功时，模型可能只是几行文本的指针，不能用于推理；GitHub 的源码 ZIP 也不能保证包含实际权重。模型下载需要网络及远端 LFS 访问权限，可能受仓库 LFS 配额限制。

当前主流程使用以下权重：

```text
models/
├── rfdetr-seg-2xlarge.b4.trt11.fp16.engine
├── rfdetr-seg-2xlarge.b4.mixed-fp16.onnx
├── rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
├── reid/mobilenet_v2-b0353104.pth
└── insightface/models/buffalo_l/
    ├── det_10g.onnx
    └── w600k_r50.onnx
```

`rfdetr.backend: auto` 按 TensorRT、ONNX Runtime 顺序选择后端。TensorRT 引擎受 GPU、CUDA、TensorRT 版本约束；不兼容时可将后端设为 `onnx`，但仍需兼容的 ONNX Runtime GPU/CUDA 运行库。不能保证任意机器直接复用本地引擎。

`models/hoop_yolo.pt` 也随模型目录纳入版本控制，但当前感知主流程未使用它；RF-DETR 模型旁的 `.json` 文件为模型元数据。

已验证的环境为 Python 3.12 + NVIDIA GPU；依赖列表见 [requirements.txt](requirements.txt)。另建环境时在自己可写的位置创建虚拟环境，安装与机器驱动兼容的 PyTorch/CUDA 后，再运行 `python -m pip install -r requirements.txt`，并将其目录用于上文 `<ENV_PREFIX>`。该文件不是跨平台锁定环境；本次没有重新安装推理依赖或验证全新环境安装。

`third_party/` 已包含所需 MMPose 源码，无需另装 MMPose；FFmpeg 优先使用系统程序，否则使用 `imageio-ffmpeg`。检查 GPU：

```bash
python -c "import torch, onnxruntime; print(torch.cuda.is_available(), onnxruntime.get_available_providers())"
```

预期 PyTorch 返回 `True`，ONNX Runtime 含 `CUDAExecutionProvider`。

## 输出与 ID 颜色

在指定的输出根目录内：

| 位置 | 内容 |
|---|---|
| `poses/poses_3d.json` | 2D/3D 骨架、全局 ID、地面坐标、篮球及质量信息 |
| `poses/metrics.json` | 推理帧数、后端、数量和耗时 |
| `poses/view*_rfdetr_pose.mp4` | 实例轮廓、2D 骨架、ID 和篮球 |
| `poses/tracks/*.jsonl` | 球员/篮球逐帧结构化记录 |
| `trajectory_pipeline/<序号>/traj_gen/` | 轨迹 JSON、原视角视频、俯视视频、额外平滑 JSON 和图片 |
| `skeletons_3d/` | 多观察角度 3D 骨架 MP4 和 GIF 预览 |

人物分割轮廓、俯视图和 3D 骨架使用同一 `player_colors` 映射：默认 ID 1–6 对应红、绿、蓝、黄、品红、青。配置是 RGB，OpenCV 绘制时转换为 BGR。颜色对应同一次运行的跟踪 ID，并不证明跟踪器永不换 ID，也不代表跨次运行的真实人员身份相同。

完整字段、坐标、缺失值和下游读取示例见 [接口文档](docs/interfaces.md)。

## 测试

在当前环境、本目录执行：

```bash
python -m pytest tests -q
```

测试临时文件随上文 `TMPDIR` 保留在本项目下。测试覆盖几何精度、轨迹异常点及非零相机偏移对齐；测试环境另需 `pytest`。
