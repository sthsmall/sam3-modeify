# build_sam3_video_predictor 处理流程文档

## 概述

`build_sam3_video_predictor` 是 SAM3 视频预测器的入口函数，用于创建一个支持多 GPU 的视频目标跟踪预测器。该函数返回一个 `Sam3VideoPredictorMultiGPU` 实例，提供了完整的视频分割和目标跟踪功能。

## 函数定义

```python
def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
```

**位置**: [model_builder.py:823-827](file:///mnt/d/projects/specific/sam3/sam3/model_builder.py#L823-L827)

## 类继承结构

```
Sam3BasePredictor (基类)
    └── Sam3VideoPredictor
            └── Sam3VideoPredictorMultiGPU
```

## 处理流程详解

### 1. 初始化阶段 (Sam3VideoPredictorMultiGPU.__init__)

**位置**: [sam3_video_predictor.py:102-150](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_video_predictor.py#L102-L150)

```
┌─────────────────────────────────────────────────────────────┐
│                    初始化流程                                │
├─────────────────────────────────────────────────────────────┤
│  1. 确定 GPU 配置                                           │
│     ├── 未指定 gpus_to_use → 使用当前 GPU                   │
│     └── 指定 gpus_to_use → 使用指定的 GPU 列表              │
│                                                             │
│  2. 设置分布式环境变量                                       │
│     ├── MASTER_ADDR = "localhost"                           │
│     ├── MASTER_PORT = 自动查找空闲端口                       │
│     ├── RANK = "0"                                          │
│     └── WORLD_SIZE = GPU 数量                               │
│                                                             │
│  3. 加载模型 (调用父类 __init__)                            │
│     └── build_sam3_video_model() → 构建完整模型             │
│                                                             │
│  4. 启动工作进程 (多 GPU 场景)                               │
│     ├── _start_worker_processes()                           │
│     └── _start_nccl_process_group()                         │
└─────────────────────────────────────────────────────────────┘
```

### 2. 模型构建阶段 (build_sam3_video_model)

**位置**: [model_builder.py:696-819](file:///mnt/d/projects/specific/sam3/sam3/model_builder.py#L696-L819)

```
┌─────────────────────────────────────────────────────────────┐
│                  模型构建流程                                │
├─────────────────────────────────────────────────────────────┤
│  1. 构建 Tracker 模块                                       │
│     ├── _create_tracker_maskmem_backbone() → 内存编码器     │
│     ├── _create_tracker_transformer() → Transformer         │
│     └── build_tracker() → Sam3TrackerPredictor              │
│                                                             │
│  2. 构建 Detector 模块                                      │
│     ├── _create_vision_backbone() → ViT + Neck              │
│     ├── _create_text_encoder() → 文本编码器                 │
│     ├── SAM3VLBackbone() → 视觉-语言骨干                    │
│     ├── _create_sam3_transformer() → Transformer            │
│     ├── _create_segmentation_head() → 分割头                │
│     ├── _create_geometry_encoder() → 几何编码器             │
│     └── Sam3ImageOnVideoMultiGPU() → 检测器                 │
│                                                             │
│  3. 构建主模型                                              │
│     └── Sam3VideoInferenceWithInstanceInteractivity         │
│         ├── detector: 检测器模块                            │
│         ├── tracker: 跟踪器模块                             │
│         └── 各种阈值和参数配置                              │
│                                                             │
│  4. 加载检查点                                              │
│     ├── 从 HuggingFace 下载 (默认)                          │
│     └── 从本地路径加载                                       │
└─────────────────────────────────────────────────────────────┘
```

### 3. 请求处理流程 (handle_request / handle_stream_request)

**位置**: [sam3_base_predictor.py:36-68](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_base_predictor.py#L36-L68)

```
┌─────────────────────────────────────────────────────────────┐
│                   请求处理流程                               │
├─────────────────────────────────────────────────────────────┤
│  handle_request(request)                                    │
│  ├── type="start_session" → start_session()                 │
│  ├── type="add_prompt" → add_prompt()                       │
│  ├── type="remove_object" → remove_object()                 │
│  ├── type="reset_session" → reset_session()                 │
│  ├── type="cancel_propagation" → cancel_propagation()       │
│  └── type="close_session" → close_session()                 │
│                                                             │
│  handle_stream_request(request)                             │
│  └── type="propagate_in_video" → propagate_in_video()       │
└─────────────────────────────────────────────────────────────┘
```

### 4. 会话管理流程

**位置**: [sam3_base_predictor.py:70-93](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_base_predictor.py#L70-L93)

```
┌─────────────────────────────────────────────────────────────┐
│                   会话管理流程                               │
├─────────────────────────────────────────────────────────────┤
│  start_session(resource_path, session_id)                   │
│  ├── 加载视频/图像资源                                       │
│  ├── 初始化 inference_state                                 │
│  ├── 生成或使用指定的 session_id                            │
│  └── 存储到 _all_inference_states                           │
│                                                             │
│  close_session(session_id)                                  │
│  ├── 从 _all_inference_states 移除                          │
│  └── 执行垃圾回收                                           │
│                                                             │
│  reset_session(session_id)                                  │
│  └── 调用 model.reset_state() 重置状态                      │
└─────────────────────────────────────────────────────────────┘
```

### 5. 添加提示流程 (add_prompt)

**位置**: [sam3_base_predictor.py:95-140](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_base_predictor.py#L95-L140)

```
┌─────────────────────────────────────────────────────────────┐
│                   添加提示流程                               │
├─────────────────────────────────────────────────────────────┤
│  add_prompt(session_id, frame_idx, ...)                     │
│  │                                                          │
│  ├── 输入参数:                                              │
│  │   ├── text: 文本提示                                     │
│  │   ├── points: 点提示坐标                                 │
│  │   ├── point_labels: 点提示标签                           │
│  │   ├── bounding_boxes: 边界框提示                         │
│  │   └── obj_id: 对象 ID (可选)                             │
│  │                                                          │
│  ├── 数据转换:                                              │
│  │   └── 列表 → torch.Tensor                                │
│  │                                                          │
│  └── 调用 model.add_prompt() 执行实际添加                   │
│      └── 返回 frame_idx 和 outputs                          │
└─────────────────────────────────────────────────────────────┘
```

### 6. 视频传播流程 (propagate_in_video)

**位置**: [sam3_video_inference.py:980-1100](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_video_inference.py#L980-L1100)

```
┌─────────────────────────────────────────────────────────────┐
│                  视频传播流程                                │
├─────────────────────────────────────────────────────────────┤
│  propagate_in_video(inference_state, ...)                   │
│  │                                                          │
│  ├── 1. 解析传播类型:                                       │
│  │   ├── propagation_full: 完整 VG 传播                     │
│  │   ├── propagation_partial: 部分 Tracker 传播             │
│  │   └── propagation_fetch: 直接获取已有预测                │
│  │                                                          │
│  ├── 2. 完整传播 (propagation_full):                        │
│  │   └── 调用父类 propagate_in_video()                      │
│  │                                                          │
│  ├── 3. 部分传播 (propagation_partial):                     │
│  │   ├── 获取 Tracker 推理状态                              │
│  │   ├── 逐帧传播:                                          │
│  │   │   ├── _prepare_backbone_feats()                      │
│  │   │   ├── _propogate_tracker_one_frame_local_gpu()       │
│  │   │   └── 广播结果到所有 GPU                             │
│  │   └── 构建输出并 yield 结果                              │
│  │                                                          │
│  └── 4. 直接获取 (propagation_fetch):                       │
│      └── 从 cached_frame_outputs 获取已有结果               │
└─────────────────────────────────────────────────────────────┘
```

### 7. 多 GPU 协作流程

**位置**: [sam3_video_predictor.py:152-180](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_video_predictor.py#L152-L180)

```
┌─────────────────────────────────────────────────────────────┐
│                  多 GPU 协作流程                             │
├─────────────────────────────────────────────────────────────┤
│  主进程 (rank=0)                                            │
│  ├── 接收请求                                               │
│  ├── 分发请求到所有工作进程                                  │
│  │   └── command_queues[rank].put((request, ...))          │
│  ├── 执行本地处理                                           │
│  ├── 等待所有进程完成 (torch.distributed.barrier)           │
│  └── 返回结果                                               │
│                                                             │
│  工作进程 (rank>0)                                          │
│  ├── 监听命令队列                                           │
│  ├── 执行接收到的请求                                       │
│  ├── 同步屏障等待                                           │
│  └── 循环等待下一个命令                                      │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件说明

### Sam3VideoInferenceWithInstanceInteractivity

**位置**: [sam3_video_inference.py:965](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_video_inference.py#L965)

主模型类，包含两个核心子模块：

| 组件 | 类型 | 功能 |
|------|------|------|
| detector | Sam3ImageOnVideoMultiGPU | 视觉-语言检测器，处理文本和视觉提示 |
| tracker | Sam3TrackerPredictor | 目标跟踪器，处理时序传播 |

### 关键参数配置

```python
Sam3VideoInferenceWithInstanceInteractivity(
    score_threshold_detection=0.5,    # 检测分数阈值
    assoc_iou_thresh=0.1,             # 关联 IoU 阈值
    det_nms_thresh=0.1,               # 检测 NMS 阈值
    new_det_thresh=0.7,               # 新检测阈值
    hotstart_delay=15,                # 热启动延迟帧数
    max_trk_keep_alive=30,            # 跟踪器最大保持帧数
    image_size=1008,                  # 输入图像尺寸
)
```

## 使用示例

```python
from sam3.model_builder import build_sam3_video_predictor

# 创建预测器
predictor = build_sam3_video_predictor(
    gpus_to_use=[0, 1],  # 使用 GPU 0 和 1
)

# 开始会话
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/path/to/video.mp4",
})
session_id = response["session_id"]

# 添加提示
predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "a person",
    "bounding_boxes": [[100, 100, 200, 200]],
})

# 传播到整个视频
for result in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
}):
    frame_idx = result["frame_index"]
    outputs = result["outputs"]
    # 处理每帧结果

# 关闭会话
predictor.handle_request({
    "type": "close_session",
    "session_id": session_id,
})

# 关闭预测器
predictor.shutdown()
```

## 相关文件

| 文件 | 说明 |
|------|------|
| [model_builder.py](file:///mnt/d/projects/specific/sam3/sam3/model_builder.py) | 模型构建入口 |
| [sam3_video_predictor.py](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_video_predictor.py) | 多 GPU 预测器实现 |
| [sam3_base_predictor.py](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_base_predictor.py) | 基础预测器类 |
| [sam3_video_inference.py](file:///mnt/d/projects/specific/sam3/sam3/model/sam3_video_inference.py) | 视频推理核心逻辑 |
