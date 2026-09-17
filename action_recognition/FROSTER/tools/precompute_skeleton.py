#!/usr/bin/env python3
"""
离线预处理：为视频中的每个人画骨架并保存为新视频。

推荐方案（离线处理）的理由：
  1. 训练时不需要额外计算姿态估计，速度快 → 训练不拖慢
  2. 骨架结果可以反复查看，有问题（人太多、漏检）可以人工修正
  3. 训练可复现性好：每次训练看到的骨架完全一致
  4. 可以调参（骨架颜色粗细、pose 模型大小）而不影响训练流程

缺点是需要额外的磁盘空间和一次预处理时间。

使用方式：
  # 处理 spacejam 训练集 + 验证集（默认使用 yolov8n-pose，速度快）
  python tools/precompute_skeleton.py \
      --data-dir data/spacejam \
      --pose-model yolov8n-pose.pt \
      --suffix _skeleton

  # 处理完后，训练时加 --use-skeleton 即可：
  bash tools/dist_train.sh 4 --use-skeleton
"""

import argparse
import csv
import json as json
import os
from pathlib import Path
import sys as sys
import time

import cv2
import numpy as np

# COCO 17 个关键点的骨架连接关系 (0-indexed)
COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # 头: nose -> eyes, eyes -> ears
    (5, 6),  # 肩膀连接
    (5, 7),
    (7, 9),  # 左臂: shoulder -> elbow -> wrist
    (6, 8),
    (8, 10),  # 右臂: shoulder -> elbow -> wrist
    (5, 11),
    (6, 12),  # 躯干: shoulder -> hip
    (11, 12),  # 髋部连接
    (11, 13),
    (13, 15),  # 左腿: hip -> knee -> ankle
    (12, 14),
    (14, 16),  # 右腿: hip -> knee -> ankle
]

# 关节关键点颜色 (BGR) - 统一黄色
COCO_KPT_COLORS = [
    (0, 255, 255),  # 0 nose
    (0, 255, 255),  # 1 left_eye
    (0, 255, 255),  # 2 right_eye
    (0, 255, 255),  # 3 left_ear
    (0, 255, 255),  # 4 right_ear
    (0, 255, 255),  # 5 left_shoulder
    (0, 255, 255),  # 6 right_shoulder
    (0, 255, 255),  # 7 left_elbow
    (0, 255, 255),  # 8 right_elbow
    (0, 255, 255),  # 9 left_wrist
    (0, 255, 255),  # 10 right_wrist
    (0, 255, 255),  # 11 left_hip
    (0, 255, 255),  # 12 right_hip
    (0, 255, 255),  # 13 left_knee
    (0, 255, 255),  # 14 right_knee
    (0, 255, 255),  # 15 left_ankle
    (0, 255, 255),  # 16 right_ankle
]

# 骨架连线颜色 (BGR) - 统一纯绿色
COCO_SKELETON_COLORS = [
    (0, 255, 0),  # 头
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),  # 肩膀
    (0, 255, 0),  # 左臂
    (0, 255, 0),
    (0, 255, 0),  # 右臂
    (0, 255, 0),
    (0, 255, 0),  # 左躯干
    (0, 255, 0),  # 右躯干
    (0, 255, 0),  # 髋
    (0, 255, 0),  # 左腿
    (0, 255, 0),
    (0, 255, 0),  # 右腿
    (0, 255, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线为视频绘制人体骨架，生成可供训练使用的骨架视频目录"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/spacejam",
        help="FROSTER 数据目录，包含 train.csv / val.csv",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_skeleton",
        help="输出目录后缀，例如 data/spacejam -> data/spacejam_skeleton",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="/data/ljy23/project/pose/pose/model/yolo11x-pose.pt",
        help="Ultralytics YOLO pose 模型权重路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备 cuda / cpu",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="姿态估计置信度阈值",
    )
    parser.add_argument(
        "--draw-mode",
        type=str,
        choices=["overlay", "skeleton_only", "skeleton_on_black"],
        default="overlay",
        help=(
            "绘制模式:\n"
            "  overlay            - 在原始帧上叠加骨架(推荐)\n"
            "  skeleton_only      - 只保留骨架线条，原图淡化为背景\n"
            "  skeleton_on_black  - 纯黑背景 + 彩色骨架(纯模态融合实验用)"
        ),
    )
    parser.add_argument(
        "--line-thick",
        type=int,
        default=1,
        help="骨架线条粗细",
    )
    parser.add_argument(
        "--kpt-radius",
        type=int,
        default=2,
        help="关键点圆点半径",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=1,
        help="最多画几个人，0 表示画所有检测到的人",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="多进程数（0/1 单进程；>1 使用进程池）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的骨架视频",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="仅处理前 x% 视频，用于快速调试",
    )
    return parser.parse_args()


def draw_skeleton_on_frame(
    frame, results, line_thick=3, kpt_radius=4, draw_mode="overlay", max_persons=0
):
    """在单帧上绘制骨架。
    results: ultralytics YOLO pose 的 results[0] 对象
    """
    H, W = frame.shape[:2]

    if draw_mode == "skeleton_on_black":
        canvas = np.zeros_like(frame)
    elif draw_mode == "skeleton_only":
        canvas = frame.copy()
        canvas = cv2.addWeighted(canvas, 0.15, np.zeros_like(canvas), 0.85, 0)
    else:  # overlay
        canvas = frame.copy()

    if results is None or results[0].keypoints is None:
        return canvas

    kpts_all = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
    conf_all = results[0].keypoints.conf.cpu().numpy()  # (N, 17)

    N = kpts_all.shape[0]
    if max_persons > 0:
        N = min(N, max_persons)

    for pi in range(N):
        kpts = kpts_all[pi]  # (17, 2)
        confs = conf_all[pi]  # (17,)

        # 先画骨架线
        for li, (a, b) in enumerate(COCO_SKELETON):
            if a >= len(kpts) or b >= len(kpts):
                continue
            if confs[a] < 0.3 or confs[b] < 0.3:
                continue
            xa, ya = int(kpts[a][0]), int(kpts[a][1])
            xb, yb = int(kpts[b][0]), int(kpts[b][1])
            if xa <= 0 and ya <= 0 or xb <= 0 and yb <= 0:
                continue
            color = (
                COCO_SKELETON_COLORS[li]
                if li < len(COCO_SKELETON_COLORS)
                else (255, 255, 0)
            )
            cv2.line(
                canvas, (xa, ya), (xb, yb), color, line_thick, lineType=cv2.LINE_AA
            )

        # 再画关键点
        for ki, (x, y) in enumerate(kpts):
            if confs[ki] < 0.3:
                continue
            xi, yi = int(x), int(y)
            if xi <= 0 and yi <= 0:
                continue
            color = (
                COCO_KPT_COLORS[ki] if ki < len(COCO_KPT_COLORS) else (255, 255, 255)
            )
            cv2.circle(canvas, (xi, yi), kpt_radius, color, -1, lineType=cv2.LINE_AA)

    return canvas


def process_one_video(src_path, dst_path, pose_model, args):
    """处理单个视频：逐帧读 -> 姿态估计 -> 画骨架 -> 写新视频"""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  [SKIP] 无法打开: {src_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        # 尝试另一种编码
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter.fourcc(*"avc1"), fps, (W, H))
        if not writer.isOpened():
            print(f"  [SKIP] 无法创建 writer: {dst_path}")
            cap.release()
            return False

    ok_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # YOLO pose 推理
            results = pose_model(frame, verbose=False, conf=args.conf, classes=[0])
            canvas = draw_skeleton_on_frame(
                frame,
                results,
                line_thick=args.line_thick,
                kpt_radius=args.kpt_radius,
                draw_mode=args.draw_mode,
                max_persons=args.max_persons,
            )
            writer.write(canvas)
            ok_frames += 1
    finally:
        cap.release()
        writer.release()

    if ok_frames == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    os.replace(tmp_path, dst_path)
    return True


def collect_video_paths(data_dir):
    """读取 train.csv 和 val.csv，收集 (src, relpath, label) 列表"""
    items = []
    seen = set()
    for split in ("train", "val"):
        csv_path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] 找不到 {csv_path}，跳过 {split}")
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) < 2:
                    continue
                src = row[0].strip()
                label = row[1].strip()
                if not src or src in seen:
                    continue
                seen.add(src)
                items.append({"src": src, "split": split, "label": label})
    return items


def build_output_csv(src_data_dir, dst_data_dir, items):
    """将原 train/val.csv 中的 src 路径替换为骨架视频路径，写新 CSV
    规则: 去掉原始 PREFIX，然后拼到 dst_data_dir 下。
    这里采取简单策略：每条 src 是绝对或相对路径，
      dst = dst_data_dir / <src 的相对最后两段>
    更好的方案：保留目录结构，把根目录替换。
    """
    for split in ("train", "val"):
        src_csv = os.path.join(src_data_dir, f"{split}.csv")
        dst_csv = os.path.join(dst_data_dir, f"{split}.csv")
        if not os.path.exists(src_csv):
            continue
        lines = []
        with open(src_csv, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) < 2:
                    continue
                p = row[0]
                label = row[1]
                # 尝试在 dst_data_dir 下保持相同目录结构
                # 约定：对于路径 /xxx/orig_dir/sub/A.mp4，输出到 /xxx/new_dir/sub/A.mp4
                # 策略：提取相对于 PATH_PREFIX 的相对路径。这里 PATH_PREFIX 未知，
                # 所以我们选择一个简单规则：取最后两级目录保留，其他都去掉。
                parts = Path(p).parts
                if len(parts) >= 2:
                    rel = os.path.join(*parts[-2:])
                else:
                    rel = os.path.basename(p)
                new_p = os.path.join(dst_data_dir, rel)
                lines.append(f"{new_p} {label}")
        os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
        with open(dst_csv, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[CSV] 写入 {dst_csv}: {len(lines)} 行")


def main():
    args = parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    dst_data_dir = args.data_dir.rstrip("/") + args.suffix
    os.makedirs(dst_data_dir, exist_ok=True)

    print("=" * 70)
    print("FROSTER 离线骨架预处理")
    print(f"  数据目录  : {args.data_dir}")
    print(f"  输出目录  : {dst_data_dir}")
    print(f"  Pose 模型 : {args.pose_model}")
    print(f"  设备      : {args.device}")
    print(f"  绘制模式  : {args.draw_mode}")
    print(f"  多进程数  : {args.workers}")
    print("=" * 70)

    # 加载 pose 模型
    print("\n[1/3] 加载姿态估计模型...")
    from ultralytics import YOLO

    pose_model = YOLO(args.pose_model)
    pose_model.to(args.device)

    # 收集视频列表
    print("\n[2/3] 收集视频列表...")
    items = collect_video_paths(args.data_dir)
    print(f"  共 {len(items)} 个唯一视频")
    if 0 < args.sample_ratio < 1.0:
        n = max(1, int(len(items) * args.sample_ratio))
        items = items[:n]
        print(f"  调试模式：只处理前 {len(items)} 个")

    # 计算每个视频的目标输出路径
    def compute_dst(src):
        parts = Path(src).parts
        if len(parts) >= 2:
            rel = os.path.join(*parts[-2:])
        else:
            rel = os.path.basename(src)
        return os.path.join(dst_data_dir, rel)

    tasks = []
    skip_cnt = 0
    for it in items:
        dst = compute_dst(it["src"])
        if os.path.exists(dst) and os.path.getsize(dst) > 0 and not args.overwrite:
            skip_cnt += 1
            continue
        tasks.append((it["src"], dst))
    print(f"  跳过已存在: {skip_cnt}, 待处理: {len(tasks)}")

    # 处理视频
    print(f"\n[3/3] 处理视频 ({len(tasks)} 个)...")
    t0 = time.time()
    done, fail = 0, 0
    if args.workers <= 1:
        for i, (src, dst) in enumerate(tasks):
            if (i + 1) % 50 == 0 or i == 0 or i == len(tasks) - 1:
                elapsed = time.time() - t0
                eta = (elapsed / (i + 1) * (len(tasks) - (i + 1))) if (i + 1) > 0 else 0
                print(
                    f"  [{i + 1}/{len(tasks)}] OK={done} FAIL={fail} "
                    f"elapsed={elapsed:.0f}s ETA={eta:.0f}s | {os.path.basename(src)}"
                )
            try:
                ok = process_one_video(src, dst, pose_model, args)
                if ok:
                    done += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"  [ERR] {src}: {e}")
    else:
        from multiprocessing import Pool

        # 每个进程单独加载模型会比较慢，这里退化为分块传递全局进程池内部重加载
        # 为了简单我们使用单进程加载 + imap_unordered；实际视频IO是瓶颈，
        # 多进程收益有限，用户如果指定 workers>1 我们也给一个实现。
        def _worker(task):
            s, d = task
            try:
                # 在子进程中加载模型（每个子进程一次）
                if not hasattr(_worker, "m"):
                    from ultralytics import YOLO

                    setattr(_worker, "m", YOLO(args.pose_model))
                    getattr(_worker, "m").to(args.device)
                return process_one_video(s, d, getattr(_worker, "m"), args), s, d
            except Exception as e:
                return False, s, str(e)

        with Pool(processes=args.workers) as pool:
            for i, (ok, s, info) in enumerate(pool.imap_unordered(_worker, tasks)):
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (len(tasks) - (i + 1))
                    print(
                        f"  [{i + 1}/{len(tasks)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                    )
                if ok:
                    done += 1
                else:
                    fail += 1
                    print(f"  [ERR] {s}: {info}")

    elapsed = time.time() - t0
    print(f"\n完成! 成功={done}, 失败={fail}, 总耗时={elapsed:.1f}s")

    # 重建输出 CSV
    print("\n重建 train.csv / val.csv ...")
    build_output_csv(args.data_dir, dst_data_dir, items)

    # 复制 index_label_mapping.json 和 class_weights.json
    for fn in (
        "index_label_mapping.json",
        "index_label_mapping_enhance.json",
        "class_weights.json",
    ):
        sp = os.path.join(args.data_dir, fn)
        if os.path.exists(sp):
            import shutil

            shutil.copy2(sp, os.path.join(dst_data_dir, fn))
            print(f"[COPY] {fn}")

    print("\n" + "=" * 70)
    print("全部完成！训练时使用:")
    print("  python tools/train_spacejam.py --use-skeleton \\")
    print(f"      --skeleton-suffix {args.suffix}")
    print("  或:")
    print("  bash tools/dist_train.sh 4 --use-skeleton")
    print("=" * 70)


if __name__ == "__main__":
    main()
