#!/usr/bin/env python3
"""
Open-MeDe 多人动作识别脚本 (YOLO检测 + 滑动窗口)

流程:
1. 使用YOLOv8检测视频中每帧的人物
2. 使用追踪算法关联跨帧的同一个人
3. 以检测框中心为中心扩展成正方形矩形
4. 缩放到模型需要的尺寸(224x224)
5. 对每个球员使用滑动窗口进行Open-MeDe动作识别
6. 可视化每个球员在每个窗口的动作标签

运行方式:
python openmede_yolo_sliding_window.py \
    --video /path/to/video.mp4 \
    --checkpoint /path/to/openmede_checkpoint.pyth \
    --config configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_spacejam.yaml
"""

import argparse
import cv2
import numpy as np
import os
import sys
import torch
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.config.custom_config import add_custom_config
from slowfast.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open-MeDe Multi-Person Sliding Window Recognition"
    )
    parser.add_argument("--video", type=str, default='/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4', help="视频文件路径")
    parser.add_argument(
        "--checkpoint", type=str, default='/data/ljy23/project/vlm/Open-MeDe/checkpoints/expand_FOMAML_spacejam/vitb16_8x16/checkpoints/checkpoint_epoch_00002.pyth', help="模型权重路径 (.pyth 文件)"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_spacejam.yaml",
        help="配置文件路径"
    )
    parser.add_argument("--start-frame", type=int, default=900, help="起始帧索引")
    parser.add_argument("--end-frame", type=int, default=1200, help="结束帧索引(-1到末尾)")
    parser.add_argument("--yolo-model", type=str,
        default="/data/ljy23/project/motion/NBAction/Yolo-Model/yolov8n.pt", help="YOLO模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--out-filename", type=str, default="", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=15, help="输出视频帧率")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="YOLO置信度阈值")
    parser.add_argument("--expand-ratio", type=float, default=1.7, help="检测框扩展比例")
    parser.add_argument("--input-size", type=int, default=224, help="模型输入尺寸")
    parser.add_argument("--padding-mode", type=bool,default=True, help="使用填充模式")
    parser.add_argument("--window-len", type=int, default=32, help="滑动窗口帧数")
    parser.add_argument("--stride", type=int, default=14, help="滑动步长")
    parser.add_argument("--label-map", type=str, default="", help="标签映射JSON路径")
    parser.add_argument("--save-clips", action="store_true", help="保存模型输入片段")
    parser.add_argument("--logit-scale-fix", type=float, default=None,
        help="修复logit_scale (如训练时被破坏, 设为14.29对应0.07初始值)")
    return parser.parse_args()


def load_openmede_model(config_path, checkpoint_path, device="cuda",
                        logit_scale_fix=None):
    """加载Open-MeDe模型"""
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(config_path)
    cfg = assert_and_infer_cfg(cfg)
    cfg.NUM_GPUS = 0

    print("构建Open-MeDe模型...")
    model = build_model(cfg)

    # 重建动态分类器 (确保使用spacejam标签映射)
    print("重建动态分类器...")
    model.update_state()

    print(f"加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint.get("model_state", checkpoint)
    state_dict = model.state_dict()

    if "module" in list(state_dict.keys())[0]:
        new_checkpoint_model = {}
        for key, value in checkpoint_model.items():
            new_checkpoint_model["module." + key] = value
        checkpoint_model = new_checkpoint_model

    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) <= 20:
        for k in missing:
            print(f"  Missing: {k}")

    # 修复logit_scale (可选)
    if logit_scale_fix is not None:
        with torch.no_grad():
            model.model.logit_scale.fill_(np.log(1.0 / logit_scale_fix))
        print(f"logit_scale 修复为: {model.model.logit_scale.item():.4f} "
              f"(exp={model.model.logit_scale.exp().item():.4f})")

    model = model.to(device)
    model.eval()

    # 打印模型关键信息
    print(f"\n模型类型: {type(model).__name__}")
    print(f"NUM_FRAMES: {cfg.DATA.NUM_FRAMES}")
    print(f"INPUT_SIZE: {cfg.DATA.TRAIN_CROP_SIZE}")
    print(f"NUM_CLASSES: {cfg.MODEL.NUM_CLASSES}")
    print(f"logit_scale: {model.model.logit_scale.item():.4f} "
          f"(exp={model.model.logit_scale.exp().item():.4f})")

    num_frames = cfg.DATA.NUM_FRAMES
    input_size = cfg.DATA.TRAIN_CROP_SIZE
    return model, cfg, num_frames, input_size


def load_video_frames(video_path, start_frame=0, end_frame=-1):
    """加载视频的指定帧范围"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame < 0:
        end_frame = total_frames - 1
    end_frame = min(end_frame, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"视频总帧数: {total_frames}, 加载: [{start_frame}, {end_frame}], 实际: {len(frames)} 帧")
    return frames


def bbox_to_square(bbox, frame_shape):
    """将检测框转为正方形框, 保持中心不变"""
    x1, y1, x2, y2 = bbox
    fh, fw = frame_shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)
    nx1 = cx - side / 2.0
    nx2 = cx + side / 2.0
    ny1 = cy - side / 2.0
    ny2 = cy + side / 2.0
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(fw, nx2)
    ny2 = min(fh, ny2)
    return [int(nx1), int(ny1), int(nx2), int(ny2)]


def detect_persons_yolo(frame, yolo_model, conf_thres=0.5):
    """使用YOLOv8检测帧中的人物"""
    results = yolo_model(frame, conf=conf_thres, classes=[0])
    persons = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            persons.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
    return persons


def expand_bbox(bbox, expand_ratio, frame_shape):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    nw = w * expand_ratio
    nh = h * expand_ratio
    nx1 = cx - nw / 2
    nx2 = cx + nw / 2
    ny1 = cy - nh / 2
    ny2 = cy + nh / 2
    expanded = [nx1, ny1, nx2, ny2]
    return bbox_to_square(expanded, frame_shape)


def crop_and_resize(frame, bbox, target_size=224):
    """裁剪检测框区域并缩放到目标尺寸"""
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.float32)
    return cv2.resize(cropped, (target_size, target_size))


def crop_and_resize_with_padding(frame, bbox, target_size=224, expand_ratio=1.2):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    nw, nh = w * expand_ratio, h * expand_ratio
    temp_bbox = [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]
    sq_bbox = bbox_to_square(temp_bbox, frame.shape)
    x1s, y1s, x2s, y2s = sq_bbox
    cropped = frame[y1s:y2s, x1s:x2s]
    if cropped.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.float32)
    return cv2.resize(cropped, (target_size, target_size))


def prepare_video_clip(frames, num_frames, input_size, device="cuda"):
    """将帧序列准备为模型输入格式 (1, C, T, H, W)"""
    from torchvision.transforms import Normalize

    normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )

    total = len(frames)
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = list(range(total)) + [total - 1] * (num_frames - total)

    tensors = []
    for idx in indices:
        f = frames[idx]
        t = torch.from_numpy(f).float() / 255.0
        t = t.permute(2, 0, 1)
        t = normalize(t)
        tensors.append(t)

    video_tensor = torch.stack(tensors, dim=0)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = video_tensor.unsqueeze(0).to(device)
    return video_tensor


def inference_video_clip(model, video_tensor):
    """对单个视频片段进行推理, 返回概率分布和top1预测"""
    with torch.no_grad():
        # Open-MeDe模型接受 [tensor] 作为输入 (list of pathways)
        pred = model([video_tensor])
        if isinstance(pred, list):
            pred = pred[0]
        probs = torch.softmax(pred, dim=-1)
        probs = probs.cpu().numpy().squeeze()
        top1_label = int(np.argmax(probs))
        top1_prob = float(probs[top1_label])
    return probs, top1_label, top1_prob


def create_sliding_windows(total_frames, window_len=32, stride=10):
    """创建滑动窗口"""
    windows = []
    start = 0
    idx = 0
    while start + window_len <= total_frames:
        windows.append({
            "window_idx": idx,
            "start_frame": start,
            "end_frame": start + window_len - 1,
            "frame_indices": list(range(start, start + window_len)),
        })
        start += stride
        idx += 1
    return windows


def track_persons(all_frame_detections, max_distance=100):
    """使用简单的追踪算法关联跨帧的同一个人"""
    tracked = []
    total = len(all_frame_detections)

    for fi, detections in enumerate(all_frame_detections):
        if fi == 0:
            for i, det in enumerate(detections):
                tracked.append({
                    "person_idx": i,
                    "bboxes": [None] * total,
                    "centers": [None] * total,
                })
                tracked[i]["bboxes"][fi] = det["bbox"]
                tracked[i]["centers"][fi] = det["center"]
        else:
            unmatched = list(range(len(detections)))
            cost = np.full((len(tracked), len(detections)), float("inf"))

            for ti, tp in enumerate(tracked):
                last_c = None
                for lb in range(1, min(11, fi + 1)):
                    if tp["centers"][fi - lb] is not None:
                        last_c = tp["centers"][fi - lb]
                        break
                if last_c is None:
                    continue
                for di, det in enumerate(detections):
                    d = np.sqrt((last_c[0] - det["center"][0])**2 +
                                (last_c[1] - det["center"][1])**2)
                    cost[ti, di] = d

            while unmatched:
                min_cost, min_ti, min_di = float("inf"), -1, -1
                for ti in range(len(tracked)):
                    for di in unmatched:
                        if cost[ti, di] < min_cost:
                            min_cost, min_ti, min_di = cost[ti, di], ti, di

                if min_cost < max_distance:
                    tracked[min_ti]["bboxes"][fi] = detections[min_di]["bbox"]
                    tracked[min_ti]["centers"][fi] = detections[min_di]["center"]
                    unmatched.remove(min_di)
                else:
                    break

            for di in unmatched:
                new_p = {
                    "person_idx": len(tracked),
                    "bboxes": [None] * total,
                    "centers": [None] * total,
                }
                new_p["bboxes"][fi] = detections[di]["bbox"]
                new_p["centers"][fi] = detections[di]["center"]
                tracked.append(new_p)

    for tp in tracked:
        tp["valid_frames"] = [i for i, b in enumerate(tp["bboxes"]) if b is not None]
        tp["num_valid_frames"] = len(tp["valid_frames"])

    return tracked


def visualize_results(frames, persons_windows, tracked_persons, labels, output_path, fps=30):
    """可视化多人滑动窗口动作识别结果"""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
        (255, 165, 0), (128, 0, 128), (0, 128, 128),
        (128, 128, 0), (255, 192, 203), (0, 255, 127),
    ]

    for frame_idx, frame in enumerate(frames):
        for pidx, pwindows in enumerate(persons_windows):
            current_window = None
            for w in pwindows:
                if w["start_frame"] <= frame_idx <= w["end_frame"]:
                    current_window = w
                    break

            current_bbox = None
            if pidx < len(tracked_persons) and frame_idx < len(tracked_persons[pidx]["bboxes"]):
                current_bbox = tracked_persons[pidx]["bboxes"][frame_idx]

            if current_bbox is not None and current_window is not None:
                x1, y1, x2, y2 = current_bbox
                action_name = labels[current_window["top1_label"]]
                prob = current_window["top1_prob"]
                color = colors[pidx % len(colors)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"P{pidx}: {action_name} ({prob:.1%})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = y1 - 10 if y1 > 20 else y2 + 20
                cv2.rectangle(frame, (x1, ly - th - 5), (x1 + tw + 5, ly + 5), color, -1)
                cv2.putText(frame, text, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"可视化视频已保存: {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Open-MeDe 多人动作识别 (滑动窗口版)")
    print("=" * 60)
    print(f"视频: {args.video}")
    print(f"检查点: {args.checkpoint}")
    print(f"设备: {args.device}")

    model, cfg, num_frames, input_size = load_openmede_model(
        args.config, args.checkpoint, device=args.device,
        logit_scale_fix=args.logit_scale_fix,
    )

    # 加载标签映射
    if args.label_map and os.path.exists(args.label_map):
        with open(args.label_map) as f:
            label_mapping = json.load(f)
    else:
        mapping_path = cfg.DATA.INDEX_LABEL_MAPPING_FILE
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                label_mapping = json.load(f)
        else:
            raise FileNotFoundError(f"标签映射文件未找到: {mapping_path}")

    # 提取冒号前的动作名用于可视化
    labels = [label_mapping[str(i)].split(':')[0].strip() for i in range(len(label_mapping))]
    print(f"\n动作标签 ({len(labels)} 类): {labels}")

    print(f"\n初始化YOLO模型...")
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_model)

    print(f"\n加载视频...")
    frames = load_video_frames(args.video, args.start_frame, args.end_frame)

    print(f"\n检测每帧人物...")
    all_detections = []
    for fi, frame in enumerate(frames):
        persons = detect_persons_yolo(frame, yolo_model, args.conf_thres)
        all_detections.append(persons)
        if (fi + 1) % 60 == 0:
            print(f"  已检测 {fi + 1}/{len(frames)} 帧")

    print(f"\n追踪人物...")
    tracked = track_persons(all_detections)
    print(f"追踪到 {len(tracked)} 个球员")

    tracked = [tp for tp in tracked if tp["num_valid_frames"] >= args.window_len]
    print(f"有效球员: {len(tracked)} 个")

    windows = create_sliding_windows(len(frames), args.window_len, args.stride)
    print(f"创建 {len(windows)} 个滑动窗口")

    clips_dir = None
    if args.save_clips:
        clips_dir = os.path.join(
            os.path.dirname(args.out_filename or args.video),
            "openmede_model_input_clips",
        )
        os.makedirs(clips_dir, exist_ok=True)

    # Open-MeDe的采样间隔: window_len // num_frames
    sample_interval = max(1, args.window_len // num_frames)
    print(f"\n对每个球员进行滑动窗口推理 (采样间隔: {sample_interval})...")

    persons_windows = []
    for pidx, person in enumerate(tracked):
        person_results = []

        for window in windows:
            window_frames = []
            sampled = window["frame_indices"][::sample_interval]

            for fi in sampled:
                if fi < len(person["bboxes"]) and person["bboxes"][fi] is not None:
                    bbox = person["bboxes"][fi]
                    if args.padding_mode:
                        cropped = crop_and_resize_with_padding(
                            frames[fi], bbox, input_size
                        )
                    else:
                        exp_bbox = expand_bbox(bbox, args.expand_ratio, frames[0].shape)
                        cropped = crop_and_resize(frames[fi], exp_bbox, input_size)
                    window_frames.append(cropped)

            if len(window_frames) >= num_frames:
                video_tensor = prepare_video_clip(
                    window_frames, num_frames, input_size, args.device
                )
                probs, top1_label, top1_prob = inference_video_clip(
                    model, video_tensor
                )

                person_results.append({
                    "window_idx": window["window_idx"],
                    "start_frame": window["start_frame"],
                    "end_frame": window["end_frame"],
                    "top1_label": top1_label,
                    "top1_prob": top1_prob,
                    "probs": probs.tolist(),
                })

                if clips_dir:
                    label_name = labels[top1_label]
                    clip_name = f"w{window['window_idx']:04d}_p{pidx:02d}_{label_name}_{top1_prob:.2f}.mp4"
                    clip_path = os.path.join(clips_dir, clip_name)
                    h, w = window_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(clip_path, fourcc, 30, (w, h))
                    for frm in window_frames:
                        out.write(frm)
                    out.release()

        if person_results:
            persons_windows.append(person_results)
            action_dist = {}
            for w in person_results:
                an = labels[w["top1_label"]]
                action_dist[an] = action_dist.get(an, 0) + 1
            print(f"\n  球员 {pidx}: 检测帧 {person['num_valid_frames']}/{len(frames)}")
            for act, cnt in sorted(action_dist.items(), key=lambda x: -x[1]):
                pct = cnt / len(person_results) * 100
                print(f"    {act}: {cnt} 窗口 ({pct:.1f}%)")

    print(f"\n" + "=" * 60)
    print("推理结果统计")
    print("=" * 60)
    all_counts = {}
    for pr in persons_windows:
        for w in pr:
            an = labels[w["top1_label"]]
            all_counts[an] = all_counts.get(an, 0) + 1
    total_w = sum(len(pw) for pw in persons_windows)
    for act, cnt in sorted(all_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_w * 100 if total_w else 0
        print(f"  {act}: {cnt} 窗口 ({pct:.1f}%)")

    output_path = args.out_filename
    if not output_path:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        fs = f"_f{args.start_frame}_{args.end_frame if args.end_frame >= 0 else 'end'}"
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "output", "spacejam"
        )
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{video_name}{fs}_openmede.mp4")

    print(f"\n生成可视化视频: {output_path}")
    visualize_results(frames, persons_windows, tracked, labels, output_path, args.fps)

    print("\n完成!")


if __name__ == "__main__":
    main()
