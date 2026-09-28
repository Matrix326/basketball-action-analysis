# 篮球视觉规则高光系统

将上游 `perception` 的人物 ID、关键点、球轨迹和多视角几何结果转换成篮球事件、个人统计及 MP4 高光。检测使用视觉证据与规则，不训练新模型。主要入口为 `python -m cv_rule_highlights auto-run`。

## 代码结构

实现集中在 `src/`，按处理职责分组；根目录只保留包声明和三个命令入口。

```text
cv_rule_highlights/
├── __init__.py
├── __main__.py                  # 主命令入口
├── evaluate_annotations.py      # 离线评估入口
├── rim_audit.py                 # 篮筐复核入口
├── src/
│   ├── core/                   # 输入适配、配置、帧区间、场地几何
│   ├── perception/             # 投篮视觉证据、球补检、篮筐复核
│   ├── events/                 # 控球、队伍、匿名身份和事件规则
│   ├── editing/                # 片段边界、选镜头、分类剪辑和渲染
│   ├── reporting/              # 统计、HTML 报告、审计和评估
│   └── runners/                # 命令分发、全流程编排
├── config/                     # 输入路径、标定和规则配置
├── tests/                      # 规则及流程回归测试
├── docs/                       # 运行与维护说明
├── requirements.txt
└── output/                     # 本地运行产物，不纳入 Git
```

命令保持不变。直接使用内部 Python 模块时，导入路径改为例如 `cv_rule_highlights.src.core.data`；内部旧路径 `cv_rule_highlights.data` 不再保留。

## 运行第一段

从仓库根目录执行，输出必须是尚不存在的新目录：

```bash
cd /data/wangyt/basketball-action-analysis
HIGHLIGHTS_PYTHON=/data/wangyt/envs/rfdetr-gpu/bin/python
HIGHLIGHTS_RUN=cv_rule_highlights/output/segment_1_delivery_new

"$HIGHLIGHTS_PYTHON" -m cv_rule_highlights auto-run \
  --config cv_rule_highlights/config/actual_11_9_segment_1_no_training.yaml \
  --output "$HIGHLIGHTS_RUN" \
  --segments segment_1 \
  --reuse-perception
```

完整运行生成事件分类、统计、全局高光、六名匿名球员的个人高光、事件分类视频和候选核查页面。只需分析和剪辑计划时，在上述命令末尾加 `--skip-render`。

`--reuse-perception` 禁止补跑上游全流程。篮筐扫描、ROI 球补检是独立阶段：扫描缓存或球证据缺失时仍可能重新扫描或调用 GPU。当前第一段配置已指定这些缓存。部署时必须提供配置引用的视频、感知结果、标定和缓存；它们不会随代码提交。

具体输入、每个命令的用法、输出位置和故障排查见 **[运行说明](docs/运行说明.md)**。

## 测试

运行环境另外需要 `pytest`；运行产物和临时文件都放在本项目内：

```bash
PYTHONDONTWRITEBYTECODE=1 "$HIGHLIGHTS_PYTHON" -m pytest cv_rule_highlights/tests -q \
  -o cache_dir=cv_rule_highlights/output/pytest_cache \
  --basetemp=cv_rule_highlights/output/pytest_tmp
```

当前主要输出支持进球、投失、篮板、突破/过人和视觉助攻链。抢断、盖帽的规则与候选核查保留，但效果不足，暂不作为可信统计。指标受限于人工标注完整性，不能把精选高光标注当作全部事件真值。

原始视频 `/data/tt` 只读。当前只对第一段做回归；第二、三段需要另行验证输入、标定及检测效果。

