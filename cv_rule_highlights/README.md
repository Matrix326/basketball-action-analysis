# Highlights

这是基于 `perception` 输出的最小规则引擎。当前实现检测投篮候选、篮筐穿越证据、命中/投失/未知结果，并生成可解释事件、EDL 和 MP4 片段。它不会修改感知结果，也不会把未知结果强行标为投失。

从仓库根目录运行：

```bash
python -m highlights detect \
  --config highlights/config/demo.yaml \
  --output highlights/output/run
```

生成 `events.json` 和 `edit_decision_list.json`。只剪确认命中：

```bash
python -m highlights run \
  --config highlights/config/demo.yaml \
  --output highlights/output/run \
  --size 1280 720
```

审核模式会把 `unknown` 和 `missed` 候选也剪出，适合调阈值：

```bash
python -m highlights run \
  --config highlights/config/demo.yaml \
  --output highlights/output/review \
  --review
```

配置中的 `rim` 是每个视角原始图像坐标下的 `[中心 x, 中心 y, 宽, 高]`。生产使用前必须替换为实际视频的篮圈标定；`frame_zero` 表示该源视频第 0 帧对应的同步帧号。球队映射使用感知输出中的字符串 track ID：

```yaml
teams:
  "1": red
  "2": blue
```

运行测试：

```bash
python -m pytest highlights/tests -q
```
