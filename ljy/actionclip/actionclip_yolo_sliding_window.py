"""
ActionCLIP多人动作识别脚本（滑动窗口版，每帧检测框）

流程:
1. 使用YOLOv8检测视频中每帧的人物
2. 使用追踪算法关联跨帧的同一个人
3. 以检测框中心为中心扩展成正方形矩形
4. 缩放到模型需要的尺寸(224x224)
5. 对每个球员使用滑动窗口进行ActionCLIP动作识别
6. 可视化每个球员在每个窗口的动作标签（检测框随球员移动）

标签: shooting, passing, dribbling, walking

运行方式:
python actionclip_yolo_sliding_window.py --video /data/ljy23/project/stal/clip_1200_1800.mp4
"""

import argparse
import cv2
import numpy as np
import torch
import clip
import os
from collections import defaultdict

from models.load import init_actionclip
from mmaction.utils import register_all_modules

register_all_modules(True)


def parse_args():
    parser = argparse.ArgumentParser(description='ActionCLIP Multi-Person Sliding Window Recognition')
    parser.add_argument('--video', type=str, default='/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4', help='视频文件路径')
    parser.add_argument('--start-frame', type=int, default=1200, help='起始帧索引（包含，0-based）')
    parser.add_argument('--end-frame', type=int, default=1800, help='结束帧索引（包含，-1表示到视频末尾）')
    parser.add_argument('--model', type=str, default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/work_dirs/actionclip_spacejam/20260810_102900/epoch_5.pth', help='ActionCLIP模型')
    parser.add_argument('--yolo-model', type=str, default='/data/ljy23/project/motion/NBAction/Yolo-Model/yolov8n.pt', help='YOLO模型')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--out-filename', type=str, default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/work_dirs/actionclip_spacejam/20260810_102900/A1.mp4', help='输出视频文件名（为空则自动根据视频名和帧范围生成）')
    parser.add_argument('--fps', type=int, default=15, help='输出帧率')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='YOLO置信度阈值')
    parser.add_argument('--expand-ratio', type=float, default=1.5, help='检测框扩展比例')
    parser.add_argument('--input-size', type=int, default=224, help='模型输入尺寸')
    parser.add_argument('--padding-mode', action='store_true', help='使用填充模式而不是裁剪模式')
    parser.add_argument('--window-len', type=int, default=32, help='滑动窗口帧数')
    parser.add_argument('--stride', type=int, default=14, help='滑动步长')
    parser.add_argument('--template', type=str, default='The basketball player is {}', help='文本模板')
    parser.add_argument('--use-detailed-descriptions', action='store_true',
                        help='使用详细动作描述而不是简单类别名')
    parser.add_argument('--action-descriptions', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/data/spacejam/action_descriptions.txt',
                        help='详细动作描述文件路径')
    parser.add_argument('--label-map', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/data/spacejam/label_map.txt',
                        help='标签映射文件路径')
    return parser.parse_args()


def load_video_frames(video_path, start_frame=0, end_frame=-1):
    """加载视频的指定帧范围
    
    Args:
        video_path: 视频文件路径
        start_frame: 起始帧索引（包含，0-based）
        end_frame: 结束帧索引（包含，-1表示到视频末尾）
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame < 0:
        end_frame = total_frames - 1
    end_frame = min(end_frame, total_frames - 1)
    
    # 跳到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"  视频总帧数: {total_frames}, 加载范围: [{start_frame}, {end_frame}], 实际加载: {len(frames)} 帧")
    return frames


def detect_persons_yolo(frame, yolo_model, conf_thres=0.5):
    """使用YOLOv8检测帧中的人物"""
    results = yolo_model(frame, conf=conf_thres, classes=[0])
    
    persons = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            persons.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'width': x2 - x1,
                'height': y2 - y1
            })
    
    return persons


def expand_bbox(bbox, expand_ratio, frame_shape):
    """以检测框中心为中心扩展成正方形"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    max_dim = max(width, height) * expand_ratio
    
    new_x1 = center_x - max_dim / 2
    new_y1 = center_y - max_dim / 2
    new_x2 = center_x + max_dim / 2
    new_y2 = center_y + max_dim / 2
    
    h, w = frame_shape[:2]
    new_x1 = max(0, int(new_x1))
    new_y1 = max(0, int(new_y1))
    new_x2 = min(w, int(new_x2))
    new_y2 = min(h, int(new_y2))
    
    return [new_x1, new_y1, new_x2, new_y2]


def crop_and_resize(frame, bbox, target_size=224):
    """裁剪检测框区域并缩放到目标尺寸"""
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    cropped_resized = cv2.resize(cropped, (target_size, target_size))
    # 不在这里标准化，返回 uint8 图像
    return cropped_resized


def crop_and_resize_with_padding(frame, bbox, target_size=224, expand_ratio=1.2):
    """
    使用填充模式处理检测框（先等比放大，再左右补0，保持高度）

    Args:
        frame: 原始帧
        bbox: [x1, y1, x2, y2] 原始检测框
        target_size: 目标尺寸
        expand_ratio: 放大比例（默认1.2）

    Returns:
        padded_resized: 填充并缩放后的图像（左右补黑色）
    """
    x1, y1, x2, y2 = bbox
    
    # 等比放大检测框
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    new_width = width * expand_ratio
    new_height = height * expand_ratio
    
    new_x1 = max(0, int(center_x - new_width / 2))
    new_y1 = max(0, int(center_y - new_height / 2))
    new_x2 = min(frame.shape[1], int(center_x + new_width / 2))
    new_y2 = min(frame.shape[0], int(center_y + new_height / 2))

    # 提取放大后的检测框区域
    cropped = frame[new_y1:new_y2, new_x1:new_x2]

    if cropped.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.float32)

    cropped_height = new_y2 - new_y1
    cropped_width = new_x2 - new_x1

    # 创建正方形画布（高度与放大后检测框高度相同）
    square_size = cropped_height
    canvas = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    # 计算填充位置（居中放置）
    x_offset = (square_size - cropped_width) // 2

    # 将裁剪区域放置到画布上（左右补0）
    if x_offset >= 0:
        canvas[:, x_offset:x_offset+cropped_width] = cropped
    else:
        # 如果检测框宽度超过高度，裁剪两侧
        crop_x_start = (-x_offset)
        canvas = cropped[:, crop_x_start:crop_x_start+square_size]

    # 缩放到目标尺寸
    padded_resized = cv2.resize(canvas, (target_size, target_size))
    # 不在这里标准化，返回 uint8 图像
    return padded_resized


def save_person_clip_to_temp(person_frames, temp_path, fps=30):
    """将球员的帧序列保存为临时视频"""
    height, width = person_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in person_frames:
        out.write(frame)
    
    out.release()


def inference_person_clip(model, person_frames, text_features, device='cuda',
                           save_clip=False, output_path=None, fps=30):
    """对单个球员的帧序列进行推理

    Args:
        model: ActionCLIP模型
        person_frames: 帧序列（已经过crop和resize的224x224图像）
        text_features: 文本特征
        device: 设备
        save_clip: 是否保存视频片段
        output_path: 保存路径（如果save_clip为True）
        fps: 输出视频帧率
    """
    import torch
    import numpy as np
    from torchvision.transforms import Normalize

    # CLIP的标准化参数
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711))

    # 将帧序列转换为tensor
    # person_frames: list of [H, W, C] uint8 images
    # 转换为 [T, C, H, W] float32 tensor
    frames_tensor = []
    for frame in person_frames:
        # uint8 [0, 255] -> float32 [0, 1]
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        # [H, W, C] -> [C, H, W]
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frames_tensor.append(frame_tensor)

    # Stack成 [T, C, H, W]
    video_tensor = torch.stack(frames_tensor, dim=0)  # [T, C, H, W]

    # 标准化
    video_tensor = normalize(video_tensor)

    # 添加batch维度 [1, T, C, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        video_features = model.encode_video(video_tensor)

    video_features /= video_features.norm(dim=-1, keepdim=True)
    similarity = (100 * video_features @ text_features.T).softmax(dim=-1)
    probs = similarity.cpu().numpy().squeeze()

    top1_label = int(np.argmax(probs))
    top1_prob = float(probs[top1_label])

    # 保存处理后的视频片段（如果需要）
    if save_clip and output_path:
        # person_frames已经是224x224的图像，直接保存
        height, width = person_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in person_frames:
            out.write(frame)

        out.release()

    return probs, top1_label, top1_prob


def create_sliding_windows(total_frames, window_len=32, stride=10):
    """创建滑动窗口"""
    windows = []
    window_idx = 0
    
    start_idx = 0
    while start_idx + window_len <= total_frames:
        windows.append({
            'window_idx': window_idx,
            'start_frame': start_idx,
            'end_frame': start_idx + window_len - 1,
            'frame_indices': list(range(start_idx, start_idx + window_len))
        })
        start_idx += stride
        window_idx += 1
    
    return windows


def track_persons(all_frame_detections, max_distance=100):
    """
    使用简单的追踪算法关联跨帧的同一个人
    
    Args:
        all_frame_detections: list of list, 每帧的检测结果
        max_distance: 最大匹配距离
    
    Returns:
        tracked_persons: list of dict, 每个球员包含每帧的检测框（None表示未检测到）
    """
    tracked_persons = []
    total_frames = len(all_frame_detections)
    
    for frame_idx, detections in enumerate(all_frame_detections):
        if frame_idx == 0:
            # 第一帧：初始化所有检测到的人
            for i, det in enumerate(detections):
                tracked_persons.append({
                    'person_idx': i,
                    'bboxes': [None] * total_frames,
                    'centers': [None] * total_frames
                })
                tracked_persons[i]['bboxes'][frame_idx] = det['bbox']
                tracked_persons[i]['centers'][frame_idx] = det['center']
        else:
            # 后续帧：使用匈牙利算法匹配
            unmatched_detections = list(range(len(detections)))
            
            # 计算代价矩阵
            cost_matrix = np.full((len(tracked_persons), len(detections)), float('inf'))
            
            for ti, tp in enumerate(tracked_persons):
                # 查找最后一个有效的中心位置（往前看最多10帧）
                last_center = None
                for lookback in range(1, min(11, frame_idx + 1)):
                    if tp['centers'][frame_idx - lookback] is not None:
                        last_center = tp['centers'][frame_idx - lookback]
                        break
                
                if last_center is None:
                    continue
                
                for di, det in enumerate(detections):
                    dx = abs(last_center[0] - det['center'][0])
                    dy = abs(last_center[1] - det['center'][1])
                    dist = np.sqrt(dx*dx + dy*dy)
                    cost_matrix[ti, di] = dist
            
            # 贪婪匹配
            while unmatched_detections:
                # 找到最小代价
                min_cost = float('inf')
                min_ti, min_di = -1, -1
                
                for ti in range(len(tracked_persons)):
                    for di in unmatched_detections:
                        if cost_matrix[ti, di] < min_cost:
                            min_cost = cost_matrix[ti, di]
                            min_ti, min_di = ti, di
                
                if min_cost < max_distance:
                    # 匹配成功
                    tracked_persons[min_ti]['bboxes'][frame_idx] = detections[min_di]['bbox']
                    tracked_persons[min_ti]['centers'][frame_idx] = detections[min_di]['center']
                    unmatched_detections.remove(min_di)
                else:
                    # 没有更多匹配
                    break
            
            # 未匹配的检测框创建新的人
            for di in unmatched_detections:
                new_person = {
                    'person_idx': len(tracked_persons),
                    'bboxes': [None] * total_frames,
                    'centers': [None] * total_frames
                }
                new_person['bboxes'][frame_idx] = detections[di]['bbox']
                new_person['centers'][frame_idx] = detections[di]['center']
                tracked_persons.append(new_person)
    
    # 统计有效帧数
    for tp in tracked_persons:
        tp['valid_frames'] = [i for i, b in enumerate(tp['bboxes']) if b is not None]
        tp['num_valid_frames'] = len(tp['valid_frames'])
    
    return tracked_persons


def visualize_person_actions_sliding(frames, persons_windows, tracked_persons, labels, output_path, fps=30):
    """
    可视化多人滑动窗口动作识别结果
    
    Args:
        frames: 原始视频帧列表
        persons_windows: 每个球员的窗口推理结果
        tracked_persons: 追踪数据（包含每帧的检测框）
        labels: 动作标签列表
        output_path: 输出视频路径
        fps: 输出帧率
    """
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    person_colors = [
        (255, 80, 80),    # 红色
        (80, 255, 80),    # 绿色
        (80, 80, 255),    # 蓝色
        (255, 255, 80),   # 黄色
        (255, 80, 255),   # 紫色
        (80, 255, 255),   # 青色
        (255, 165, 0),    # 橙色
        (128, 0, 128),    # 紫色
        (0, 128, 128),    # 深青色
        (128, 128, 0),    # 橄榄色
        (255, 192, 203),  # 粉色
        (0, 255, 127),    # 春绿色
        (75, 0, 130),     # 靛青色
        (255, 215, 0),    # 金色
    ]
    
    for frame_idx, frame in enumerate(frames):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for person_idx, person_windows in enumerate(persons_windows):
            # 1. 找到当前帧对应的动作窗口
            current_window = None
            for window in person_windows:
                if window['start_frame'] <= frame_idx <= window['end_frame']:
                    current_window = window
                    break
            
            # 2. 获取当前帧的检测框（从追踪数据）
            current_bbox = None
            if person_idx < len(tracked_persons):
                if frame_idx < len(tracked_persons[person_idx]['bboxes']):
                    current_bbox = tracked_persons[person_idx]['bboxes'][frame_idx]
            
            # 3. 如果有检测框且在窗口内，则可视化
            if current_bbox is not None and current_window is not None:
                # 使用原始检测框（不扩展）
                x1, y1, x2, y2 = current_bbox
                
                # 使用当前窗口的动作标签
                action_name = labels[current_window['top1_label']]
                prob = current_window['top1_prob']
                color = person_colors[person_idx % len(person_colors)]
                
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制动作标签
                text = f"P{person_idx}: {action_name} ({prob:.1%})"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                label_y = y1 - 10 if y1 > 20 else y2 + 20
                cv2.rectangle(frame, (x1, label_y - text_h - 5), (x1 + text_w + 5, label_y + 5), color, -1)
                cv2.putText(frame, text, (x1 + 2, label_y), font, font_scale, (255, 255, 255), thickness)
        
        out.write(frame)
    
    out.release()
    print(f"可视化视频已保存: {output_path}")


def main():
    args = parse_args()
    
    print("="*60)
    print("ActionCLIP 多人动作识别 (滑动窗口版，每帧检测框)")
    print("="*60)
    print(f"视频文件: {args.video}")
    print(f"帧范围: [{args.start_frame}, {args.end_frame if args.end_frame >= 0 else '末尾'}]")
    print(f"模型: {args.model}")
    print(f"YOLO模型: {args.yolo_model}")
    print(f"设备: {args.device}")
    # 初始化ActionCLIP模型
    print(f"\n初始化ActionCLIP模型...")
    model, preprocess, clip_len = init_actionclip(args.model, device=args.device)

    # 使用模型的clip_len确定采样间隔（对齐训练数据）
    # 例如: window_len=32, clip_len=8 → 每4帧抽1帧
    #       window_len=32, clip_len=16 → 每2帧抽1帧
    sample_interval = max(1, args.window_len // clip_len)
    print(f"  模型帧数 (num_adapter_segs): {clip_len}")

    print(f"\n滑动窗口参数:")
    print(f"  window_len: {args.window_len} 帧")
    print(f"  stride: {args.stride} 帧")
    print(f"  采样间隔: 每{sample_interval}帧抽1帧 → {args.window_len // sample_interval}帧给模型")
    print(f"  模型输入帧数: {clip_len} 帧")
    print(f"  检测框处理模式: {'填充模式（左右补0）' if args.padding_mode else '裁剪模式（扩展成正方形）'}")
    if not args.padding_mode:
        print(f"  检测框扩展比例: {args.expand_ratio}")
    print(f"  模型输入尺寸: {args.input_size}x{args.input_size}")
    print(f"\n动作标签:")
    with open(args.label_map, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    for i, label in enumerate(labels):
        print(f"  {i}: {label}")
    
    # 初始化YOLO模型
    print(f"\n初始化YOLO模型...")
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_model)

    # 准备文本特征
    print(f"\n准备文本特征...")
    if args.use_detailed_descriptions and os.path.exists(args.action_descriptions):
        # 使用详细描述
        print(f"使用详细动作描述: {args.action_descriptions}")
        text_descriptions = []

        # 读取详细描述文件
        label_to_desc = {}
        with open(args.action_descriptions, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    action_name, desc = line.split(':', 1)
                    label_to_desc[action_name.strip()] = desc.strip()

        # 按照标签顺序组织描述
        for label in labels:
            if label in label_to_desc:
                text_descriptions.append(label_to_desc[label])
                print(f"  {label}: {label_to_desc[label][:80]}...")
            else:
                # 如果没有详细描述，使用模板
                text_descriptions.append(args.template.format(label))
                print(f"  {label}: (使用简单描述) {args.template.format(label)}")
    else:
        # 使用简单描述
        text_descriptions = [args.template.format(label) for label in labels]
        print(f"使用简单动作描述（模板: {args.template}）")

    text = clip.tokenize(text_descriptions).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 加载视频帧
    print(f"\n加载视频...")
    frames = load_video_frames(args.video, args.start_frame, args.end_frame)
    print(f"加载完成: {len(frames)} 帧")
    
    # 使用YOLO检测每帧的人物
    print(f"\n检测每帧人物...")
    all_frame_detections = []
    
    for frame_idx, frame in enumerate(frames):
        persons = detect_persons_yolo(frame, yolo_model, args.conf_thres)
        all_frame_detections.append(persons)
        
        if (frame_idx + 1) % 60 == 0:
            print(f"  已检测 {frame_idx + 1}/{len(frames)} 帧")
    
    # 使用追踪算法关联跨帧人物
    print(f"\n追踪人物...")
    tracked_persons = track_persons(all_frame_detections)
    print(f"追踪到 {len(tracked_persons)} 个球员")
    
    # 保存原始追踪数据（用于可视化）
    original_tracked_persons = tracked_persons.copy()
    
    # 过滤掉检测框太少的球员
    tracked_persons = [tp for tp in tracked_persons if tp['num_valid_frames'] >= args.window_len]
    print(f"有效球员: {len(tracked_persons)} 个")
    
    # 创建滑动窗口
    print(f"\n创建滑动窗口...")
    windows = create_sliding_windows(len(frames), args.window_len, args.stride)
    print(f"创建 {len(windows)} 个滑动窗口")

    # 创建保存视频片段的目录
    clips_output_dir = os.path.join(os.path.dirname(args.out_filename), 'model_input_clips')
    os.makedirs(clips_output_dir, exist_ok=True)
    print(f"\n模型输入视频片段保存目录: {clips_output_dir}")

    # 为每个球员提取帧并进行滑动窗口推理
    print(f"\n对每个球员进行滑动窗口推理...")
    persons_windows = []
    
    for person_idx, person in enumerate(tracked_persons):
        person_window_results = []
        
        for window in windows:
            window_frames = []
            window_bbox = None
            valid_count = 0
            sample_count = 0
            
            # 根据模型clip_len自动计算采样间隔
            # 例如: window_len=32, clip_len=8 → 每4帧抽1帧
            sampled_indices = window['frame_indices'][::sample_interval]

            for fi in sampled_indices:
                if fi < len(person['bboxes']) and person['bboxes'][fi] is not None:
                    bbox = person['bboxes'][fi]

                    # 根据模式选择不同的处理方式
                    if args.padding_mode:
                        # 填充模式：先等比放大检测框(1.2倍)，再左右补0
                        processed_bbox = bbox
                        cropped = crop_and_resize_with_padding(frames[fi], bbox, args.input_size, expand_ratio=1.2)
                    else:
                        # 裁剪模式：扩展检测框成正方形
                        expanded_bbox = expand_bbox(bbox, args.expand_ratio, frames[0].shape)
                        processed_bbox = expanded_bbox
                        cropped = crop_and_resize(frames[fi], expanded_bbox, args.input_size)

                    if window_bbox is None:
                        window_bbox = processed_bbox

                    window_frames.append(cropped)
                    valid_count += 1
                    sample_count += 1

            if valid_count >= clip_len:  # 至少需要 clip_len 帧给模型
                # 进行推理，获取预测标签
                probs, top1_label, top1_prob = inference_person_clip(
                    model, window_frames, text_features, args.device
                )

                # 生成有意义的文件名并保存视频片段
                predicted_label = labels[top1_label]
                clip_filename = f"window{window['window_idx']:04d}_person{person_idx:02d}_{predicted_label}_prob{top1_prob:.2f}.mp4"
                clip_output_path = os.path.join(clips_output_dir, clip_filename)

                # 直接保存视频片段（不再重新推理）
                height, width = window_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_output_path, fourcc, 30, (width, height))
                for frame in window_frames:
                    out.write(frame)
                out.release()
                
                person_window_results.append({
                    'window_idx': window['window_idx'],
                    'start_frame': window['start_frame'],
                    'end_frame': window['end_frame'],
                    'bbox': window_bbox,
                    'top1_label': top1_label,
                    'top1_prob': top1_prob,
                    'probs': probs.tolist()
                })
        
        if person_window_results:
            persons_windows.append(person_window_results)
            
            action_dist = {}
            for w in person_window_results:
                action_name = labels[w['top1_label']]
                action_dist[action_name] = action_dist.get(action_name, 0) + 1
            
            print(f"\n  球员 {person_idx}:")
            print(f"    检测帧数: {person['num_valid_frames']}/{len(frames)} 帧")
            print(f"    动作分布:")
            for action, count in sorted(action_dist.items(), key=lambda x: -x[1]):
                pct = count / len(person_window_results) * 100
                print(f"      {action}: {count} 窗口 ({pct:.1f}%)")
    
    # 打印统计信息
    print(f"\n" + "="*60)
    print("推理结果统计")
    print("="*60)
    
    all_action_counts = {}
    for person_window_results in persons_windows:
        for w in person_window_results:
            action_name = labels[w['top1_label']]
            all_action_counts[action_name] = all_action_counts.get(action_name, 0) + 1
    
    total_windows = sum(len(pw) for pw in persons_windows)
    print("动作分布:")
    for action, count in sorted(all_action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_windows * 100 if total_windows else 0
        print(f"  {action}: {count} 窗口 ({pct:.1f}%)")
    
    # 可视化
    output_path = args.out_filename
    if not output_path:
        # 自动生成输出文件名
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        frame_suffix = f"_f{args.start_frame}_{args.end_frame if args.end_frame >= 0 else 'end'}"
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'work_dirs', 'actionclip_spacejam',
            f'{video_name}{frame_suffix}_output.mp4'
        )
    elif not output_path.startswith('/'):
        output_path = os.path.join('/data/ljy23/project/stal/mm/mmaction2/projects/actionclip', output_path)
    
    print(f"\n生成可视化视频...")
    visualize_person_actions_sliding(frames, persons_windows, tracked_persons, labels, output_path, args.fps)
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']="2,3,4,5"
    print(torch.cuda.is_available())
    main()