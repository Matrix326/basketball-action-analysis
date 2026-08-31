# 多视角篮球动作识别流水线

4 相机（A1/A2/B3/B4）→ 检测（人球分离 Hybrid）→ RTMPose 姿态 → 3D 重建/ReID → 球轨迹后处理 → 规则引擎 → 动作事件。

**本 README 重点记录规则引擎的完整逻辑与全部参数**（曾因代码丢失从会话记录恢复过，务必保留此文档）。

---

## 1. 系统架构

```
4 视角视频
  → RF-DETR-Seg 2XL（人，真 mask）+ 微调 RF-DETR Large（球）        [detector.py: HybridRunner]
  → RTMPose 2D 姿态 → 多视角三角化 3D（重投影误差过滤）
  → ReID 球员身份（外观 + 人脸 + mask，跨视角/时序关联）
  → ball_trajectory/（离线球轨迹后处理）                           [pipeline.py]
      outlier 剔除 → 弹跳检测 → 弹道分段 → flight/dribble 分类
      → 运动学补全 → RTS 平滑 → 状态分类（逐帧 state）
  → action_rules/（规则引擎）                                      [pipeline.py]
      持球状态机（possession）→ flight 段事件判定（shoot/layup/pass/rebound/follow_up/block）
      → 低传球补充（low_pass）→ 事件输出 actions.json
```

---

## 2. 球轨迹后处理（`src/ball_trajectory/`）

### 2.1 preprocess_observations（outlier 剔除，pipeline.py）

按顺序执行：
1. **连续高速剔除**：`speeds[i] > flight_max_speed_m_s(14)` 且前后帧也超速 → 剔除（视角组合切换的跳变）
2. **孤立尖峰**：前后帧速度均 > max_speed_m_s(30) → 剔除
3. **位移检查**：距上一有效点位移 > 2.0m 且速率 > **13m/s**（原 25，球速物理上限 ~14）→ 剔除（多视角共识的错误段也会漏网，13 是实测校准值）
4. **spike 检查**：a→c 段外 0.4m 的中间点 → 剔除
5. **高度检查**：z > 10m 或 < -0.15m → 剔除
6. **locked-on final pass**（**必须在所有拒绝之后、return 之前**）：views ≤ 2、速度 < 5、距上一有效点 > 1.5m → 剔除（三角化锁错目标）

> ⚠️ 恢复教训：locked-on 必须放在**最后**（final pass），放在中间会导致 outlier 计数不同（原版 454 vs 错误版 383）。

**outlier 帧的下游处理**（关键，防止错误位置污染）：
- preprocess 返回时 `measured_positions` 中 outlier 帧置 NaN
- pipeline 构建 positions_grid 时 **outlier 帧一律 NaN**（即使 fill 标记了 filled）
- **fill_gaps 的 flank 观测必须排除 outlier**（`clean_obs_frames/clean_obs_pos` 传给 `_fill_one_gap` 和 `_extrapolate_edges`）——否则双侧拟合被错误测量污染，插值位置错误
- **z 峰值扫描回退的拟合也必须排除 outlier**（`sel` 加 `~obs.outlier_mask`）——否则回退段拟合污染（如 11867-11874 的错误回退段导致持球人跟错）

### 2.2 弹跳检测（segmentation.py `detect_bounces`）

- 帧 z ≤ ball_radius + 0.05（近地）
- 双侧弹道拟合（左/右 6 帧）→ vz 从负转正（`vz_before < -1.0 and vz_after > 1.0`）
- 2 帧邻域去重（保留 impact 更强的）
- **`frames = obs.frame_indices[~obs.outlier_mask]`**（必须排除 outlier）

### 2.3 弹道分段（segmentation.py `detect_ballistic_segments`）

**参数**：`window_seconds=0.5, residual_threshold=0.10, min_window_observations=8, gap_bridge_frames=25, min_fit_observations=8, accel_z_range=(-13,-6)`

流程：
1. **窗口一致性**：每个观测帧开 ±7 帧窗口 → `fit_ballistic`（固定 g 线性回归）→ rms > 0.10 时用 `_ransac_trim` 剔除 → rms ≤ 0.10 → consistency=True
2. **runs**：consistency 连续帧段
3. **merged**：间隔 ≤ gap_bridge_frames(25) 的 run 尝试合并，**整段 fit rms ≤ 1.25 × residual_threshold(0.125) 才合并**
4. **段生成**（**注意缩进！以下全部在 `_split_until_fits` 循环内**）：
   ```python
   for piece in _split_at_bounces(run):        # 弹跳帧处切开
       for piece in _split_until_fits(piece):  # 递归切分直到拟合
           p0, v0, rms = fit_ballistic(piece)
           if rms > 1.25 * residual_threshold: continue
           # max-speed 检查：段内最大速度 < 1.0 m/s 丢弃（静态段）
           # free-fall gate：fit_free_g 的 a_z 检查
           segments.append(...)
   ```
5. **`_ransac_trim`**：LCG 确定性采样 48 次、3 点拟合、inlier 容差 `1.5×threshold`、min_inliers=**8**、inlier 展布 ≥ **0.3m**（拒绝静态窗口的抛物线穿越区）

6. **`_split_until_fits`**（递归切分）：
   - 停止：`rms ≤ 1.25×threshold or depth ≥ 6 or n ≤ 2×min_points`
   - **首尾裁剪**（关键！）：`bad = residuals > 1.25×threshold`，bad[0] → 裁首到第一个好帧；bad[-1] → 裁尾；裁剪后递归
   - 否则在最大残差处切两半递归

> ⚠️ 恢复教训：
> - 阈值全部是 **1.25×**（merged 验证 / segment 丢弃 / _split_until_fits 停止与裁剪），不是 2.0× 或 1.5×（8/20 13:11:54 的 sed 批量修改）
> - `min_inliers=8`（不是 5）——窗口 trim 对"双峰/缓升"窗口更严格，段边界与原版一致
> - `_split_until_fits` 循环内代码缩进必须正确（16 空格）——恢复时缩进错误导致每段只用最后一个 piece 且 n<8 碎片段通过
> - `_split_at_bounces`（弹跳帧切分）+ `_split_until_fits`（递归切分）嵌套调用

### 2.4 flight/dribble 分类（classify_ballistic_segments）

- `_apex_parameters`：模拟弹道求 apex_z 和水平位移
- `touches_floor = min_z ≤ 0.9`（弧线最低点贴地——**原版有，恢复时曾丢失**）
- **flight**：apex_z ≥ 2.0 或水平位移 ≥ 2.0
- **dribble**：`touches_floor and apex_z ≤ 1.4 and (bounce_count ≥ 1 或 手接近 ≤ 1.5m)`
- 死区（1.4 < apex < 2.0）：手接近 → dribble；bounce ≥ 2 → dribble；否则 flight
- confidence：flight = `clip(1 - rms/threshold) × coverage`

### 2.5 逐帧状态（states.py `classify_states`）

- 段内（flight/dribble）→ 直接取段类型
- held：手距球 ≤ hold_reach_m(0.45) 且速度 ≤ 1.5
- ground：z ≤ 0.15 且速度 ≤ 3.0
- 其余 unknown
- held 必须持续 ≥ 6 帧（短促的 revert unknown）
- 静态残渣（远离球员、速度 ≤ 0.05、持续 60 帧）→ unknown
- 5 帧多数过滤

> ⚠️ **没有 dribble fallback**（distance≤0.9/z≤1.2/speed≥0.8 的兜底是恢复期误加的，已删除）——运球兜底由 action_rules 的 fallback 分支负责，两处共存会吃掉 dribble_start 事件（Dribbling 22/31 → 30/31 的修复）

### 2.6 补全与平滑

- `fill_gaps`：弹道插值补洞（bounce 桥接），RTS 平滑
- 中值平滑窗口 **13**（`_median_smooth_positions`，pipeline 最后一步）
- 轨迹输出：每帧 position/velocity/state/observed/outlier/filled

### 2.7 z 峰值扫描回退（pipeline.py process_data）

detect 之后**补充**语义弧线：z > 1.5 连续段（gap ≤ 5 帧）且峰值 ≥ 2.2m，且与已有段覆盖 < 50% → 补充为 flight 段。
（注意：这是恢复期加的辅助，段起点（z>1.5）晚于真实出手——事件时间会偏后。保留它使 flight 段数接近原版 199。）

---

## 3. 规则引擎（`src/action_rules/pipeline.py`）

### 3.1 持球状态机（逐帧，for frame in grid）

- **held**（state=held 且手距球 ≤ hold_reach 0.45）：接球检测（`prev_max_dist > hold_reach` 且无前 10 帧 flight → catch + 手递手 pass）
- **dribble**（state=dribble 且手距球 ≤ dribble_reach 1.2）：连续 `possess_switch_frames(5)` 帧确认切换 handler
- **flight**：`dist_to_player > release_dist(0.6)` → release 事件；球进入他人手（`prev_d > catch_reach+0.1`）→ pending catch（2 帧后确认球停住 → catch）
- **unknown/ground fallback**：球低（z < 1.2）近人（≤ 1.2）有速度（> 0.5）连续 5 帧 → `dribble_start`（fallback）+ handler 切换
- **possession[frame] = handler**（每帧末尾赋值，flight 段事件用）

### 3.2 flight 段事件判定（for seg in flight_segments）

**段预处理**：
- flight 段相邻 gap ≤ 5 帧合并
- `_scan_launch_arcs`：z 峰值语义弧线补充（launch_min_z 1.8）
- `_extend_flight_segments`：向后扩展到 release（球 z ≥ 1.4 且速度 ≥ 2.0 且无手），向前扩展到球继续移动；重叠段合并（**只 extend 一次**）

**每段计算**：
- apex_z / horizontal_displacement / hoop_horizontal（到筐最小水平距离）/ release_dist_hoop（起点到筐）
- **release_actor**：`possession[start-1] or possession[start]` → 手存在性检查（缺失或距球 > release_dist×2.5 → None）→ 仍 None 时 **probe 前 8 帧**（最近手 ≤ release_dist×2.5）
- **catch_actor**：段内最近手（**排除 release_actor 自己的手**——自己的手是出手接触不是接球）

**判定顺序**（每个 flight 段，action 初始 None）：
1. **deflected**（段起点 z 跳变 > 0.8）：球被拍/抢 → block（防守手 ≤ 0.5m 且弧线朝筐且 apex ≥ 2.0）或 scoop pass（队友接）或 None
2. **follow_up（补篮）**：fu_actor（release_actor 或最近 catch/rebound）非空 + apex ≥ 2.0 + 起点距筐 ≤ 2.0 + 时长 ≤ 4.0s + 前一 shoot/layup 为 **miss** + gap ≤ 90 帧 + 中间无真实 dribble（fallback 和 rim-pop 误判的除外）
3. **shoot/layup**：
   ```
   (near_hoop and near_rim or release_dist_hoop ≥ three_point_radius 5.5)
   and approaches_hoop            # 弧线最近点在后 15% 之后
   and hoop_horizontal ≤ 3.5      # 弧线确实接近篮筐（防止远距离传球误判）
   and apex > 2.0
   and release_actor is not None
   and (apex ≥ 2.9               # 高弧直接算投篮
        or not (catch_actor 非空且 ≠ release 且 d ≤ 0.5)   # 被接的低弧是传球
        or near_rim
        or release_dist_hoop ≥ 5.5)
   ```
   kind 由 `_shot_kind`（单手/双手举球）决定 shoot vs layup；三分 = 弧线最远点 ≥ 5.5m
4. **pass**（action 仍 None 时）：catch_actor 非空且 ≠ release_actor 且 d ≤ catch_reach×2 且水平位移 > 0.8m
5. **layup**（action 仍 None 时）：near_hoop 且 release_actor 非空（低弧近筐无接球人）
6. **shoot/layup 的 result**：`_shot_result`（3D 穿越筐平面 3.05m 且水平 ≤ 0.6 → make；fall_through 兜底：越过筐心后从筐口落下）
7. **rebound**（miss 后）：end 后 40 帧内有人（非 release）距筐 ≤ 2.0m 接球（d ≤ catch_reach×1.5）
8. **post-segment pass**：action 仍 None 且 release_actor 非空 → end 后 12 帧内非 release 人接球 → pass

> ⚠️ 恢复教训：
> - **pass/layup 判定必须带 `action is None` 条件**——否则 follow_up/block 被 elif 链覆盖（Follow-up 3/6 → 6/6 的修复）
> - **catch 必须排除 release_actor**——出手前自己的手接触不是接球（修复"球被队友接却判 layup"）
> - **shoot 条件加 hoop_horizontal ≤ 3.5**——三分距离 + approaches_hoop 的传球弧线（球飞向 4m 外）不再误判投篮

### 3.3 低传球补充（`_low_pass_events`）

- 球 z ≤ 1.7、速度 ≥ 3.0 m/s 开始 run（非 held 状态）
- 速度 < 1.8（3.0×0.6）→ settled：有人接（d ≤ 0.9）且起点有人传（d ≤ 0.95）且不同人 → pass（low=True）
- 长度 > 60 帧 → 关闭 run

### 3.4 全部参数（config.yaml `action_rules:` 段）

| 参数 | 默认/当前值 | 含义 |
|---|---|---|
| three_point_radius_m | 5.5 | 实测三分弧半径（场地重标定 R=6.54，评估用 5.5） |
| hold_reach_m | 0.45 | 持球距离 |
| release_distance_m | 0.6 | 出手判定距离 |
| catch_reach_m | 0.5 | 接球判定距离 |
| dribble_reach_m | 1.2 | 运球判定距离 |
| possess_switch_frames | 5 | 持球切换确认帧数 |
| possess_hold_frames | 30 | 失去接触后保留持球帧数 |
| flight_min_seconds | 0.3 | 最短飞行段 |
| follow_up_max_gap_frames | 90 | 补篮窗口 |
| follow_up_max_seconds | 4.0 | 补篮最长时长（含 grab-and-raise） |
| make_fall_through_reach_m | 0.5 | 穿口判定容差 |
| pass_min_horizontal_m | 0.8 | 传球最小水平位移 |
| shoot_hoop_distance_m | 2.0 | near_hoop 阈值 |
| shoot_min_apex_m | 2.0 | 投篮最低弧顶 |
| shot_near_rim_reach_m | 1.2 | near_rim 阈值 |
| shot_approach_min_fraction | 0.15 | approaches_hoop 的最近点位置分数 |
| shot_min_rim_apex_m | 2.9 | 命中判定弧顶门槛 |
| shot_max_miss_distance_m | 3.5 | 弧线接近筐的最大距离（防传球误判投篮） |
| layup_release_dist_m | 1.6 | 上篮出手距离 |
| layup_max_seconds | 0.8 | 上篮最长时长 |
| rebound_hoop_distance_m | 2.0 | 篮板距筐阈值 |
| rebound_probe_frames | 40 | 篮板探测窗口 |
| low_pass_release_speed_m_s | 3.0 | 低传球出手速度 |
| low_pass_hand_reach_m | 0.95 | 低传球出手人手距 |
| low_pass_catch_reach_m | 0.9 | 低传球接球人距离 |
| low_pass_max_z_m | 1.7 | 低传球最大高度 |
| low_pass_max_frames | 60 | 低传球最长时长 |
| dribble_fallback_max_z | 1.2 | 兜底运球最高 z |
| dribble_fallback_min_speed | 0.5 | 兜底运球最低速度 |
| dribble_fallback_confirm_frames | 5 | 兜底运球确认帧数 |
| pass_post_segment_frames | 12 | 段后传球探测窗口 |
| pass_post_segment_reach_m | 0.8 | 段后传球接球距离 |
| flight_merge_gap_frames | 5 | flight 段合并间隔 |
| flight_extend_min_z_m | 1.4 | 段扩展最低 z |
| flight_extend_min_speed_m_s | 2.0 | 段扩展最低速度 |
| flight_extend_held_reach_m | 0.9 | 段扩展遇手停止距离 |
| deflect_z_jump_m | 0.8 | 击球判定 z 跳变 |
| block_hand_reach_m | 0.5 | 盖帽手距 |
| make_rim_height_m | 3.05 | 筐平面高度 |
| make_horizontal_reach_m | 0.6 | 命中水平容差 |
| make_fall_through_reach_m | 0.5 | 穿口水平容差 |
| make_probe_frames | 40 | 命中探测帧数 |
| hand_on_ball_reach_m | 1.35 | 投篮手型判定（双手）距离 |
| one_hand_near_m / one_hand_far_m | 0.85 / 1.25 | 单手/双手判定 |

---

## 4. 运行方法

```bash
# 1. 全流程（检测 + 姿态 + 3D + ReID + 球）
python src/run_rfdetr_full_pipeline.py --config config/config.yaml

# 2. 球轨迹后处理（输入 poses_3d.json → 输出 ball_trajectory.json）
python src/ball_trajectory/run_ball_trajectory.py --config config/config.yaml --print-stats

# 3. 规则引擎（输入 ball_trajectory.json + poses_3d.json + hoop_3d.json → actions.json）
python src/action_rules/run_action_rules.py --config config/config.yaml

# 4. GT 评估
python src/action_rules/evaluate_gt.py \
    --gt /data/ljy23/project/stal/1-3v3-action.json \
    --actions output/rfdetr_multiview/poses/actions.json \
    --id-map /data/ljy23/project/ref_ours/id_map.json \
    --start-frame 0 --end-frame 19739 --tolerance 50 --output /tmp/gt_eval.json
```

**GT 评估口径**：类型匹配（GT_TO_OURS）+ 时间容差 50 帧 + actor（id_map 映射，actor 缺失宽容）。非可比类别（Defence 等）不计入。

---

## 5. 切换场地 / 新数据指南

换一块场地（或换相机/换比赛时段）时，**按此顺序调整**：

### 5.1 场地标定（必须）

场地几何（世界系）完全由标定决定，标定错误会让篮筐 4 视角投影不一致、球 3D 跳变、三分判定失效。

1. **4 视角各取一帧**，标注参考点：篮圈中心、底线 4 角点、禁区矩形、三分弧上 3 点（参考 `project/rule/calibrate_view.py`、`joint_calib.py`）
2. 联合优化 4 参数使多视角 PnP 残差最小：**底线宽 W、篮筐到底线 D、半场深 DEPTH、三分弧半径 R**
3. 写入 `assets/extrinsic_parameters/extrinsics_new_calibration.json`（新外参）
4. 验证：篮筐 4 视角重投影误差 < 15px；球 3D 在无球时应无跳变

### 5.2 需要更新的参数（config.yaml）

| 参数 | 位置 | 说明 |
|---|---|---|
| `camera.extrinsics_path` | config | 新外参 json（4 视角） |
| `camera.court_world_bounds` | config | 场地世界系范围 `[xmin, xmax, ymin, ymax]`（当前 `[0, 13, 0, 12.5]`——半场 13×12.5m） |
| `action_rules.three_point_radius_m` | config | **三分弧实测半径**（当前 5.5——场地实测 R=6.54 但评估用 5.5；换场地需重新拟合三分弧：8 点拟合残差 ±6cm） |
| `hoop_3d.json` | output | 篮筐 3D 位置（由标定生成，`run_hoop_detection.py`） |
| `ball.exclusion_zones` | config | 各视角固定背景误检区（画面中重复被误检为球的静态物体，如护栏）——换场地需重新标注 |
| `camera.intrinsics_path` | config | 相机换了才需要（内参畸变） |
| `ball.max_distance / max_jump` | config | 球检测距离上限/单帧位移上限（不同场地尺寸/相机高度可能要调） |

### 5.3 验证流程（换场地后必跑）

1. `run_hoop_detection.py` 重新生成 hoop_3d.json
2. 跑检测 + 球轨迹，检查：球 3D 跳变率（原版 2.86% 参考）、flight 段数（约 200 段/20 分钟）
3. 跑动作规则，用一段人工标注验证三分/命中判定（三分半径、make 判定依赖场地参数）

---

## 6. 重要超参调整指南

**调参顺序建议**：先调检测阈值（球/人检出）→ 再看轨迹跳变（速度/平滑）→ 最后动作阈值（命中/三分/传球）。每次只改一个参数，用 `evaluate_gt.py` 对比。

### 6.1 检测（`rfdetr:` 段）

| 参数 | 默认 | 调整方法 | 影响 |
|---|---|---|---|
| `person_threshold` | 0.35 | 调高减少误检、调低增加召回 | 人检测数量（影响持球判定） |
| `ball_threshold` | 0.16 | 同上 | 球检出率（调高 → 少而准、调低 → 多而杂） |
| `ball_min_size` / `ball_max_size` | 5 / 100 px | 远视角球小（view3/4）可调低 min | 球框尺寸过滤 |
| `roi_polygons` | 空 | 填"0.0,0.4 1.0,0.4..."多边形排除场外区域 | 减少场外误检 |

### 6.2 球轨迹（`ball_trajectory:` 段）

| 参数 | 默认 | 调整方法 | 影响 |
|---|---|---|---|
| `residual_threshold_m` | 0.10 | 调大 → 更多段（含噪声弧）、调小 → 更少更严 | 弹道分段的松紧（**代码内阈值 = 1.25× 此值**） |
| `max_speed_m_s` | 30 | 球速物理上限，跳变多时可调低 | 孤立尖峰剔除 |
| `flight_max_speed_m_s` | 14 | 视角切换跳变剔除（真实球速 ~12-14） | 连续高速段剔除 |
| 位移检查速率阈值 | **13**（代码内） | 球速物理上限，跳变漏网时调低 | 多视角共识的错误段剔除（当前已校准） |
| `min_fit_observations` | 8 | 调大 → 段更少更完整 | 最短段长度 |
| `window_seconds` | 0.5 | 调大 → 窗口拟合更稳但边界模糊 | 一致性窗口 |
| `gap_bridge_frames` | 25 | 调大 → 更多合并 | run 合并间隔 |
| `accel_z_range_m_s2` | [-13, -6] | 投篮重力范围 | free-fall gate |
| 中值平滑窗口 | 13（代码内） | 调大更平滑但削掉快速弧线、调小保留细节 | 输出轨迹平滑（跳变率 4.3%→3.45% 是 5→13 的效果） |

**代码内参数**（改代码，无 config）：`_ransac_trim` 的 trials=48 / min_inliers=8 / inlier_tolerance=1.5 / min_spread=0.3；`_split_until_fits` 的 1.25× 阈值和 depth=6。

### 6.3 动作规则（`action_rules:` 段）

| 参数 | 默认 | 调整方法 | 影响 |
|---|---|---|---|
| `three_point_radius_m` | 5.5 | **换场地必调**（实测三分弧半径） | 三分判定（从出手点最远点到筐） |
| `hold_reach_m` | 0.45 | 调大 → 更多帧判 held（持球宽松） | 持球状态机 |
| `release_distance_m` | 0.6 | 调大 → 更早判出手 | release 事件 |
| `catch_reach_m` | 0.5 | 调大 → 更多接球判定 | catch/pass/rebound |
| `shoot_min_apex_m` | 2.0 | 调大 → 低弧不算投篮（传球更准） | shoot/layup 判定 |
| `shot_min_rim_apex_m` | 2.9 | 调大 → 高弧才算投篮 | 高弧传球 vs 投篮边界 |
| `shot_max_miss_distance_m` | 3.5 | 调大 → 更多弧线算投篮（含传球） | 弧线接近筐的距离上限 |
| `shot_approach_min_fraction` | 0.15 | 调大 → approaches_hoop 更严 | "球向筐飞"的判定 |
| `make_horizontal_reach_m` | 0.6 | 调大 → 更多命中（宽松） | 命中判定（穿筐半径） |
| `make_fall_through_reach_m` | 0.5 | 调大 → 穿口判定更宽松 | 命中兜底判定 |
| `follow_up_max_gap_frames` | 90 | 调大 → 更久之后仍算补篮 | 补篮窗口 |
| `follow_up_max_seconds` | 4.0 | 补篮含 grab-and-raise 阶段 | 补篮时长 |
| `pass_min_horizontal_m` | 0.8 | 调大 → 短传不算 pass | 传球最小位移 |
| `low_pass_*` | 3.0/0.95/0.9/1.7 | 低传球（地面传球）检测的松紧 | 低弧传球补充 |
| `rebound_hoop_distance_m` | 2.0 | 调大 → 更远也算篮板 | 篮板判定 |
| `dribble_fallback_*` | 1.2/0.5/5 | 运球兜底检测 | dribble_start 事件 |

**调参技巧**：
- **投篮类掉**（Shooting/Layup/Jump）→ 检查 `shoot_min_apex_m` / `make_horizontal_reach_m` / `shot_min_rim_apex_m`（放宽）
- **传球误判投篮**（Passing 掉）→ 调大 `shot_min_rim_apex_m` 或调小 `shot_max_miss_distance_m`（但会误伤高弧投篮——这是当前已知的固有取舍）
- **命中率不对**（make/miss 分布异常）→ 调 `make_rim_height_m`（3.05 标准）和 `make_horizontal_reach_m`
- **三分不准** → 重新拟合 `three_point_radius_m`（用 8 点弧线拟合）

---

## 7. 评估结果（GT1 = 1-3v3-action.json，0-19739 帧）

| 版本 | 匹配率 |
|---|---|
| 原版（8/26） | 93%（152/164） |
| 恢复初期 | 70%（115/164） |
| 规则修复后 | 89%（146/164） |
| **+ 跳变段修复（fill/z 回退排除 outlier）** | **90%（147/164）** |

**90% 版分类明细**：Dribbling 30/31、Follow-up 6/6、Layup 22/22、Shooting 15/15、Shooting-Jump 17/18、Three 2/2、Passing 18/26、Passing-Bounce 29/32、Rebound 8/12。

**已知短板**（多为检测层/GT 语义固有）：
- wrong_player（8 个）：多人争抢时 rebound/持球人判定（最近手不一定对）；或 id_map 映射边界
- time_offset（8 个）：pass 事件帧与 GT 出手时刻差 50-135 帧（真实传球被高弧 shoot 条件捕获，nearest pass 是无关事件）
- Rebound no_event（1 个）：球被拍出后手距球 > 2m（距筐 3-4m）不满足探测条件

### 5.1 三段 GT 测试（不同场次视频）

| 段 | 视频 | GT 文件 | id_map | 当前 | 原版 |
|---|---|---|---|---|---|
| GT1 | A1-1/A2-1/B3-1/B4-1 | 1-3v3-action.json | id_map.json | **90%**（147/164） | 93% |
| GT2 | A1-2/A2-2/B3-2/B4-2 | 2-3v3-action.json | id_map_seg2.json | **78%**（104/134） | 90% |
| GT3 | A1-3/A2-3/B3-3/B4-3 | 3-3v3-action.json | id_map_seg3.json | **80%**（73/91） | 82% |

- GT2/GT3 的视频与 GT1 不同（-2/-3 系列），检测产物在 `project/rule/poses_seg2/`、`poses_seg3/`（8/26 原版检测）
- 评估命令：`evaluate_gt.py --gt stal/2-3v3-action.json --actions <actions> --id-map ref_ours/id_map_seg2.json --tolerance 50`
- id_map 可用事件共现自动建立（每个 GT id 取 ±60 帧内同类型事件出现最多的 track）
- GT3 差距主要是 Rebound（8 miss：GT 的 Rebound ID 是争抢者集合，我们的"接球人"判定语义不同）和 time_offset
- GT2 差距主要是 Passing 事件错位（真实传球漏检，nearest pass 是无关事件）——依赖轨迹质量，规则层难修

### 7.2 传球 vs 投篮的区分（已探索，结论记录）

**量化**：GT1 的 58 个 Passing 中 15 个被误判为 shoot/layup（26%）；GT2 9 个；GT3 5 个。这是 Passing 的主要问题。

**尝试过的 5 类信号（全部验证失败，勿重复试）**：
1. **接球人（catch_d ≤ 0.5）**：投篮时篮下球员手也在 0.5m 内 → 误伤投篮（Layup 22→16）
2. **catch 持续 3 帧**：弹筐球在篮下停留，手持续接近 → 仍误伤
3. **catch 位置距筐**：吊传（高吊球）就是接到篮下（13 个误判里 8 个 catch 距筐 ≤ 2m）
4. **release 距筐 > 2.5**：远投（三分）未中弹筐被篮下接触 → 更差（78%）
5. **物理碰撞建模**（用户建议的方向）：
   - 速度方向突变：**RTS + 中值平滑（window 13）抹平了碰撞突变**——真投篮也检测不到
   - 原始观测 z 反弹/速度骤降：检测噪声 ±0.5m 掩盖单帧特征
   - 穿筐检测（z 穿过 3.05 平面，水平 0.6/1.0m）：真投篮穿筐点距筐 0.81m（重建误差内）→ 0.6 漏检、1.0 误报（传球在筐 0.88m 外下降也被检出）
   - 段末端 z：真投篮 1.4-3.3 vs 传球 1.2-2.0（重叠）

**结论**：球 3D 重建误差 ±0.3-0.5m 淹没碰撞物理特征——**当前数据精度下无法可靠区分高弧传球和投篮**。**未来可行路径**（需检测/数据层支持）：① 2D 穿筐检测（球 2D 框与篮筐 2D 投影重叠——poses_3d.json 已有 balls_2d，像素级不受重建误差影响）；② 平滑前检测碰撞（在 preprocess 后立即检测）；③ 球 3D 融合精度提升。

---

## 8. 本次代码恢复的关键差异记录（防再次丢失）

代码曾在整理上传时被误删，从会话记录（~/.claude/projects/*.jsonl 的 Read/Edit/Write 记录）恢复。恢复时容易丢失/改错的点：

1. **segmentation.py 阈值全部是 1.25×**（不是 2.0×/1.5×）——来自 8/20 13:11:54 的 sed 批量修改
2. **`_split_until_fits` 循环内缩进**（16 空格）——缩进错误会导致段生成错乱
3. **`_ransac_trim`：min_inliers=8**（不是 5）、trials=48、inlier_tolerance=1.5、min_spread=0.3
4. **`outlier_mask` 过滤**：detect_ballistic_segments 和 detect_bounces 都用 `obs.frame_indices[~obs.outlier_mask]`
5. **classify 的 touches_floor（min_z ≤ 0.9）**——dribble 判定需要
6. **states.py 没有 dribble fallback**（兜底运球在 action_rules 里）
7. **action_rules 的 pass/layup 判定带 `action is None`**——防覆盖 follow_up/block
8. **catch_actor 排除 release_actor**——出手前自己的手不是接球
9. **shoot 条件加 hoop_horizontal ≤ 3.5**——防远距离传球误判投篮
10. **approaches_hoop 是 shoot 顶层条件**（约束所有 shoot，不只三分）
11. **z 峰值扫描回退**（pipeline.py）——补充 detect 漏掉的弧线（flight 段数 217 ≈ 原版 199）
12. **min_fit_observations=8**（detect 签名默认），config `ball_trajectory.min_fit_observations: 8`

---

## 9. 模型与数据

- 人检测：RF-DETR-Seg 2XL（ONNX 768，真 mask）
- 球检测：微调 RF-DETR Large（`models/finetuned/rfdetr_large_ball_player.pth`，EMA mAP50 0.912 / AP_ball 0.630）
- 微调数据：自建 27,983 张（球 25,899 + 球员 78,358 标注）
- 姿态：RTMPose-m（body7）
- ReID：MobileNetV2 外观 + InsightFace 人脸
- 场地标定：`assets/extrinsic_parameters/extrinsics_new_calibration.json`（W=13.0 / D=1.7 / DEPTH=12.5，实测三分弧 R=6.54）
- 数据目录：`/data/ljy23/data/videodata/11.19`（A1/A2/B3/B4 四视角同步视频）
