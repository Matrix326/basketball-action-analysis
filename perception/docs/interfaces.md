# Perception 输入输出接口

## 模块边界

`perception/` 接收同步多相机视频与标定，输出统一 ID 的检测、姿态、3D 坐标和轨迹。

本目录不输出动作类别、投篮事件、命中结果或高光剪辑，也未加入动作识别适配器。

## 输入与坐标

本文沿用 README 的占位约定：`<REPO_ROOT>` 为仓库根目录，`<ENV_PREFIX>` 为 Python 环境目录，`<DATA_ROOT>` 为输入视频数据根目录。使用时替换成实际绝对路径；尖括号占位符不是配置加载器变量。环境准备和 Git LFS 模型拉取方式见 [README](../README.md#模型与环境)。

主配置为 `config/config.yaml`。路径支持 `${project_root}`、`${data_root}`。使用 `project_root: auto` 时根目录是配置文件所在目录的父目录，因此自定义配置应放在 `perception/config/` 下。自定义文件不会与默认文件自动合并；明确指定的文件应先用预检确认存在，旧加载器可能在找不到文件时回退默认配置。

| 输入 | 格式与约束 |
|---|---|
| `videos` | 视角名到视频路径的映射，默认四路 1920×1080、30 FPS 已去畸变 MP4 |
| `camera.view_to_camera` | view1/2/3/4 → A1/A2/B3/B4，名称必须存在于内外参 JSON |
| 内参 JSON | 相机名 → 对象；`K_undistorted` 为对应输入分辨率的 3×3 矩阵 |
| 外参 JSON | 相机名 → 对象；`R_w2c` 为 3×3，`t_w2c` 为长度 3；满足 `X_camera = R_w2c @ X_world + t_w2c` |
| 世界坐标 | 单位米，XY 为球场平面，Z 向上；原点和方向由标定定义，不由视频决定 |
| 像素坐标 | 原始去畸变输入图像，左上角原点，X 向右、Y 向下，不是网络缩放后的像素 |
| 球场背景 | `assets.court_background` 图片；背景布局须与 `trajectory.court_total_x/y` 及标定坐标相符 |

代码读取标定时会遍历内参中的相机，因此内参包含的相机也应提供对应外参。当前半场有效范围是 X∈[0,15]、Y∈[0,14] 米；俯视背景展示的是完整 15×28 米球场。更换场地时不能只替换视频而保留原标定或排除区域。

### 时间轴

- 命令行使用同步帧号，区间为 `[start_frame, end_frame)`。
- 视角 v 的原视频帧号为 `f_source = f_sync + camera.frame_offsets[v]`；正偏移表示向后读。
- 所有结构化结果中的帧号都是 `f_sync`；输出视频第 k 帧对应 `start_frame + k`。
- 原视频时间是 `f_source / fps`；输出片段时间是 `(f_sync - start_frame) / fps`。
- 所有视角须同 FPS 并事先同步。偏移只解决整数帧差，不校正帧率漂移或时钟误差。
- 非零偏移只应用在视频读取位置，不能再次加到 JSON 键上。

## 命令行

在 `perception/` 下执行，各脚本支持 `--help`。

| 参数 | 预检 `check_inputs.py` | 完整 `run_rfdetr_full_pipeline.py` | 分析 `rfdetr_pose_multiview.py` |
|---|---|---|---|
| `--config PATH` | 支持 | 支持 | 支持 |
| `--start-frame N` | 支持 | 支持 | 支持 |
| `--end-frame N` | 支持，结束帧不包含 | 同左 | 同左 |
| `--limit N` | 优先于结束帧 | 同左 | 同左 |
| `--views view1 view2 ...` | 至少两个不同的已配置视角 | 选择推理视角 | 选择推理视角 |
| `--output-base PATH` | 无 | 统一输出根目录 | 无 |
| `--output-dir PATH` | 无 | 无 | 分析结果直接写入此目录 |
| `--skip-analysis` | 无 | 复用已有 poses_3d.json | 无 |
| `--skip-smoothing` | 无 | 不执行额外平滑阶段 | 无 |
| `--skip-3d-animation` | 无 | 不生成 3D MP4/GIF | 无 |

默认起始帧为 `trajectory.start_frame`；默认结束帧为起始帧加 `int(process_seconds * fps)`。预检输出 JSON 报告并以 0 退出；检查失败以 1 退出。推理错误会抛出异常并以非零状态退出，输出文件不具备事务性，失败目录可能含部分结果，应换用新目录重跑。

`--skip-analysis` 不核对历史配置、视频和帧范围。必须由调用方保证复用数据匹配，且 `<output-base>/poses/poses_3d.json` 存在。

## Python 接口

从本目录启动 Python，使用同一套依赖环境：

```python
from config import load_config
from src.check_inputs import check_inputs
from src.rfdetr_pose_multiview import RFDetrPoseMultiViewPipeline

check_inputs("config/config.yaml", start_frame=900, end_frame=1200)
config = load_config("config/config.yaml")
pipeline = RFDetrPoseMultiViewPipeline(config)
result = pipeline.process(
    video_paths=config.video_paths,
    output_dir="output/python_demo/poses",
    start_frame=900,
    end_frame=1200,
)
# result 为与 poses_3d.json 对应的结果字典；同时写出分析产物。
```

这只执行分析阶段；完整轨迹与动画推荐调用完整 CLI，避免在下游复制流程编排逻辑。配置的 `override({"output.reid_3d_dir": path})` 接受点分隔键，`merge({...})` 接受嵌套字典。

底层 `create_multi_view_animation` 的 Python 参数 `end_frame` **包含结束帧**；完整 CLI 已将结束帧减 1。下游自行调用时勿混用两种约定。

## poses/poses_3d.json

`schema_version = "2.0-rfdetr-rtmpose"`。JSON 中帧号与 ID 对象键均为字符串，加载后不要按字符串字典序排序帧号。Python 直接返回的字典中这些键可能为整数。

| 字段路径 | 类型、形状与含义 |
|---|---|
| `video_info[view]` | `path,width,height,fps,total_frames,frame_offset`，源视频信息 |
| `models` | `detector,detector_backend,pose,keypoint_names` |
| `poses_3d[frame][id]` | 17×3 世界坐标，单位米；可能包含缺失关键点 |
| `poses_2d[frame][id][view].bbox` | 长度 4，原图像素 `[x1,y1,x2,y2]` |
| `...keypoints_xy` | 17×2 原图像素坐标 |
| `...keypoints_conf` | 长度 17，关键点置信度 |
| `...mask_foot_xy` | 轮廓脚底锚点，长度 2，像素 |
| `...ground_position` | 该视角地面投影，长度 3，世界米 |
| `...detection_confidence` | 人物检测置信度 |
| `ground_positions_3d[frame][id]` | 多视角融合后的球员地面位置，长度 3，世界米 |
| `balls_2d[frame][view]` | `center_xy`、`bbox`、`confidence`；像素坐标 |
| `balls_3d[frame]` | 在线滤波后的球位置，长度 3，世界米 |
| `balls_3d_predicted[frame]` | 布尔值，true 表示预测/维持位置，不是当前真实观测 |
| `quality[frame][id]` | 下述姿态质量信息 |
| `postprocessing.bone_refinement` | 骨长优化统计 |

`quality` 包含 `views`、`valid_3d_keypoints`、`raw_valid_3d_keypoints`、`predicted_3d`、`predicted_track`、`mean_reprojection_error_px`。最后一项是像素误差，不是米；应与有限值检查、观测视角数和预测标记共同用于质量过滤。

当前输出**不保存完整实例掩码**，人物轮廓主要体现在分析视频中；也不包含原始未滤波 `ball_measurements` 或标定文件快照。需要这些数据的消费者不能直接假设字段存在。

### COCO-17 顺序

`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`。

### 缺失数据与身份

轨迹和篮球观测可能缺帧；某人不一定在每帧、每视角出现，不能用数组行号代替真实帧号。同一帧可以存在某人的 2D 观测、地面轨迹和质量记录，但因有效关键点不足而没有该人的 3D 骨架；这些对象中的 ID 集合不一定相等，应按帧号和 ID 做可缺失关联。球员 ID 从 1 开始，在单次运行内用于跨视角关联，不是人工标注身份，也不保证重启后的同一数字指向同一人。

当前序列化允许 `NaN`，Python 的 `json.load` 可读取，但这不是严格 JSON 标准中的数字。对接严格 JSON 解析器前需递归把非有限数转换为 `null`，并使用 `allow_nan=False` 重新写出；不能将缺失点当成原点。

```python
import json
import numpy as np

with open("output/demo_900_1200/poses/poses_3d.json", encoding="utf-8") as stream:
    data = json.load(stream)
assert data["schema_version"] == "2.0-rfdetr-rtmpose"
for frame_key in sorted(data["poses_3d"], key=int):
    frame = int(frame_key)
    for player_id, values in data["poses_3d"][frame_key].items():
        joints = np.asarray(values, dtype=float)
        valid = np.isfinite(joints).all(axis=1)
        quality = data["quality"].get(frame_key, {}).get(player_id, {})
        if quality.get("predicted_track") or quality.get("predicted_3d"):
            continue
        # 使用 joints[valid]；保留 valid 以维持 COCO-17 索引。

observed_balls = {
    int(frame): xyz
    for frame, xyz in data["balls_3d"].items()
    if not data["balls_3d_predicted"].get(frame, False) and np.isfinite(xyz).all()
}
```

即使 `balls_3d_predicted=false`，球位置仍经过在线滤波，不是原始三角化测量。

## JSONL 与轨迹文件

`poses/tracks/player_tracks.jsonl` 每行一个球员在一帧的位置：

```json
{"frame_index": 900, "track_id": 1, "ground_xyz": [8.1, 5.4, 0.0]}
```

`poses/tracks/ball_tracks.jsonl` 按有 2D 球观测的帧写出：

```json
{"frame_index": 900, "views": {"view1": {"center_xy": [100, 200], "bbox": [90, 190, 110, 210], "confidence": 0.8}}, "world_xyz": [2.0, 2.5, 3.0]}
```

示例数值仅说明格式。3D 球位置不可用时 `world_xyz` 可为 `null`，JSONL 不含完整预测标记；需区分观测/预测时读取主 JSON。JSONL 为稀疏记录，不能假设与源视频逐行对应。

`trajectory_pipeline/<序号>/traj_gen/player_trajectory.json` 及 `smooth_traj.json` 的主要结构：

```json
{
  "final_merged_finished_trajectories": {
    "player_1": {
      "900": {"x": 8.1, "y": 5.4, "confidence": 1.0}
    }
  }
}
```

X/Y 是世界米，**不是俯视图像素**；`confidence: 1.0` 是当前导出的固定值，不是校准概率。`smooth_traj.json` 是额外跳变清理和平滑的结果；视频在该额外步骤之前生成，并不从这个文件重新渲染。

目录序号 1、2、3、4 是选定视频的遍历顺序，不是球员 ID；默认对应 view1、view2、view3、view4。各视角共享 3D/地面数据，因此其轨迹 JSON 可能重复同一世界轨迹。

## 可视化与颜色

`player_colors` 为 RGB 数组，颜色索引是 `(track_id - 1) % len(player_colors)`。默认 ID 1–6 为红、绿、蓝、黄、品红、青。分割轮廓、原视角骨架、俯视轨迹和 3D 骨架遵循这一映射；OpenCV 内部使用转换后的 BGR。篮球使用独立样式，不属于球员 ID 调色板。

完整默认四视角输出 13 个 MP4：4 个分析视频、4 个原视角轨迹视频、4 个俯视视频、1 个 3D 骨架视频，另有 GIF 预览。GIF 可抽帧，不用于帧精确交接。更改视角数量、可视化开关或动画 FPS 会改变输出数量/播放时长。
