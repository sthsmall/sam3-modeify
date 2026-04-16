# SAM3 跟踪评估

本目录包含使用 SAM3 在 AIR-aircraft 和 SAT-ship 数据集上进行目标跟踪并评估结果的脚本。

## 文件说明

- `run_sam3_viso.py` - 在数据集上运行 SAM3 并生成 MOT 格式的跟踪结果和可视化视频
- `evaluate_results.py` - 使用 TrackEval 评估跟踪结果
- `README.md` - 本说明文件

## 环境依赖

- Python 3.8+
- ultralytics
- opencv-python
- numpy
- TrackEval

安装依赖：
```bash
pip install ultralytics opencv-python numpy
pip install TrackEval
```

## 使用步骤

### 1. 运行 SAM3 生成跟踪结果

```bash
python run_sam3_viso.py
```

**功能：**
- 自动识别 AIR-aircraft 和 SAT-ship 数据集
- 根据序列名称自动确定提示词（如 AIR-aircraft 使用 "aircraft"）
- 对每个序列类型运行第一个序列进行测试
- 生成 MOT 格式的跟踪结果文件
- 生成带有边界框和轨迹的可视化视频
- 结果保存到 `results/` 目录，视频保存到 `videos/` 目录

**配置参数：**
- `DATASET_ROOT` - 数据集根目录
- `OUTPUT_DIR` - 结果输出目录
- `VIDEO_OUTPUT_DIR` - 视频输出目录
- `MODEL_PATH` - SAM3 模型路径
- 提示词自动从序列名称提取

### 2. 评估跟踪结果

```bash
python evaluate_results.py
```

**功能：**
- 使用 TrackEval 评估跟踪结果
- 计算 MOTA、IDF1、IDP、IDR、IDs、FM、MT、ML、FP、FN 等指标
- 生成详细的评估报告
- 结果保存到 `output/` 目录

**配置参数：**
- `DATASET_ROOT` - 数据集根目录
- `RESULTS_DIR` - 跟踪结果目录
- `EVAL_OUTPUT_DIR` - 评估结果输出目录

## 数据集结构

数据集使用 MOT 格式，结构如下：

```
laboratory_data/
├── AIR-aircraft/
│   └── train/
│       ├── MOT17-01-FRCNN/
│       │   ├── img1/          # 图像帧
│       │   ├── gt/            # 标注文件
│       │   │   └── gt.txt     # ground truth
│       │   └── seqinfo.ini    # 序列信息
│       ├── MOT17-02-FRCNN/
│       └── ...
├── SAT-ship/
│   └── train/
│       ├── MOT17-01-FRCNN/
│       ├── MOT17-02-FRCNN/
│       └── ...
└── VISO/
    ├── train/
    └── test/
```

## 结果格式

### 跟踪结果文件

生成的跟踪结果使用 MOT 格式，每行包含：

```
frame_id, track_id, x, y, w, h, conf, -1, -1, -1
```

- `frame_id` - 帧号
- `track_id` - 跟踪 ID
- `x, y` - 边界框左上角坐标
- `w, h` - 边界框宽高
- `conf` - 置信度
- `-1, -1, -1` - 保留字段

### 可视化视频

生成的视频包含：
- 边界框：每个目标用不同颜色的矩形框标注
- 轨迹：显示目标移动的历史轨迹线
- ID标签：显示目标ID和置信度
- 帧号：左上角显示当前帧号

## 评估指标说明

### 主要指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **MOTA** | Multi-Object Tracking Accuracy | 多目标跟踪准确度，综合考虑误检、漏检和ID切换 |
| **MOTP** | Multi-Object Tracking Precision | 多目标跟踪精度，衡量定位准确度 |
| **IDF1** | Identity F1 Score | ID F1分数，综合衡量ID识别精度 |
| **IDP** | Identity Precision | ID精确率 |
| **IDR** | Identity Recall | ID召回率 |

### 跟踪质量指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **IDs** | ID Switches | ID切换次数，越少越好 |
| **FM (Frag)** | Fragmentation | 碎片化次数，跟踪中断的次数 |
| **MT** | Mostly Tracked | 主要跟踪目标数（跟踪率>80%） |
| **ML** | Mostly Lost | 主要丢失目标数（跟踪率<20%） |

### 检测质量指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **FP** | False Positives | 误检数 |
| **FN** | False Negatives | 漏检数 |

## 评估结果示例

### AIR-aircraft 序列

| 指标 | 数值 |
|------|------|
| MOTA | 100.0 |
| MOTP | 66.305 |
| IDF1 | 100.0 |
| IDP | 100.0 |
| IDR | 100.0 |
| IDs | 0 |
| FM | 0 |
| MT | 1 |
| ML | 0 |
| FP | 0 |
| FN | 0 |

**分析：** AIR-aircraft序列表现优秀，完美跟踪了唯一的飞机目标。

### SAT-ship 序列

| 指标 | 数值 |
|------|------|
| MOTA | -257.75 |
| MOTP | 60.893 |
| IDF1 | 13.199 |
| IDP | 8.345 |
| IDR | 31.551 |
| IDs | 3 |
| FM | 6 |
| MT | 0 |
| ML | 0 |
| FP | 593 |
| FN | 73 |

**分析：** SAT-ship序列误检较多，需要调整参数以提高跟踪质量。

### AIR-ship 序列

| 指标 | 数值 |
|------|------|
| MOTA | -2.2854 |
| MOTP | 74.756 |
| IDF1 | 0.95247 |
| IDP | 15.044 |
| IDR | 0.4918 |
| IDs | 0 |
| FM | 2 |
| MT | 0 |
| ML | 1 |
| FP | 576 |
| FN | 20638 |

**分析：** AIR-ship序列漏检严重，可能是由于目标较小或与背景相似导致。

### SAT-airplane 序列

| 指标 | 数值 |
|------|------|
| MOTA | -291.94 |
| MOTP | 56.052 |
| IDF1 | 10.771 |
| IDP | 6.9731 |
| IDR | 23.656 |
| IDs | 0 |
| FM | 6 |
| MT | 0 |
| ML | 1 |
| FP | 587 |
| FN | 142 |

**分析：** SAT-airplane序列误检较多，需要调整参数以提高跟踪质量。

## 参数优化建议

### 减少误检（FP过高）
- 提高 `score_threshold_detection` 参数（默认0.3）
- 提高 `new_det_thresh` 参数（默认0.0）

### 减少漏检（FN过高）
- 降低 `score_threshold_detection` 参数
- 增加 `max_trk_keep_alive` 参数（默认60）

### 减少ID切换（IDs过高）
- 增加 `init_trk_keep_alive` 参数
- 启用 `masklet_confirmation_enable`

## 注意事项

1. 确保已安装 TrackEval：`pip install TrackEval`
2. 确保 SAM3 模型文件 `sam3.pt` 存在
3. 评估前确保已生成跟踪结果
4. 每个序列需要 `seqinfo.ini` 文件
5. 可根据需要修改提示词以适应不同目标类型

## 目录结构

```
eval/
├── run_sam3_viso.py      # 运行脚本
├── evaluate_results.py   # 评估脚本
├── README.md             # 说明文档
├── results/              # 跟踪结果目录
│   ├── AIR-aircraft/     # AIR-aircraft序列的跟踪结果
│   │   └── MOT17-01-FRCNN.txt
│   ├── SAT-ship/         # SAT-ship序列的跟踪结果
│   │   └── MOT17-01-FRCNN.txt
│   ├── AIR-aircraft_MOT17-01-FRCNN.txt  # 原始跟踪结果文件
│   └── SAT-ship_MOT17-01-FRCNN.txt      # 原始跟踪结果文件
├── videos/               # 可视化视频目录
│   ├── AIR-aircraft_MOT17-01-FRCNN_tracking.mp4
│   └── SAT-ship_MOT17-01-FRCNN_tracking.mp4
└── output/               # 评估结果目录
    ├── AIR-aircraft/     # AIR-aircraft序列的评估结果
    └── SAT-ship/         # SAT-ship序列的评估结果
```

## 结果目录组织说明

### 1. results/ 目录

**作用：** 存放SAM3生成的跟踪结果文件。

**文件结构：**
- `{sequence_type}_MOT17-01-FRCNN.txt` - 原始跟踪结果文件，使用MOT格式
- `{sequence_type}/` - 为TrackEval评估准备的目录结构
  - `MOT17-01-FRCNN.txt` - 复制到对应目录的跟踪结果文件

**MOT格式说明：**
每行包含：`frame_id, track_id, x, y, w, h, conf, -1, -1, -1`
- `frame_id` - 帧号（从1开始）
- `track_id` - 目标跟踪ID
- `x, y` - 边界框左上角坐标
- `w, h` - 边界框宽度和高度
- `conf` - 目标置信度
- `-1, -1, -1` - 保留字段

### 2. videos/ 目录

**作用：** 存放带有跟踪可视化的视频文件。

**文件说明：**
- `{sequence_type}_MOT17-01-FRCNN_tracking.mp4` - 跟踪可视化视频
  - 包含边界框、轨迹线、ID标签和帧号显示
  - 不同目标使用不同颜色区分

### 3. output/ 目录

**作用：** 存放TrackEval评估结果，按序列类型分文件夹组织。

**目录结构：**
- `{sequence_type}/` - 按序列类型分文件夹
  - `pedestrian_summary.txt` - 行人类别评估结果摘要
  - `pedestrian_detailed.csv` - 详细评估结果
  - `summary.json` - JSON格式的评估摘要

**评估结果文件说明：**

1. **pedestrian_summary.txt**
   - 包含主要评估指标的摘要
   - 第一行是指标名称，第二行是对应的数值
   - 指标包括：
     - HOTA：高阶跟踪精度
     - DetA：检测精度
     - AssA：关联精度
     - DetRe：检测召回率
     - DetPr：检测精确率
     - AssRe：关联召回率
     - AssPr：关联精确率
     - LocA：定位精度
     - OWTA：最优权重跟踪精度
     - MOTA：多目标跟踪准确度
     - MOTP：多目标跟踪精度
     - IDF1：ID F1分数
     - IDP：ID精确率
     - IDR：ID召回率
     - 以及其他详细指标如CLR_TP、CLR_FN、CLR_FP、IDSW、MT、PT、ML、Frag等

2. **pedestrian_detailed.csv**
   - 详细的评估结果，包含不同阈值下的性能指标
   - 第一行是详细的指标名称，包括每个阈值（5%到95%）下的各项指标
   - 第二行是具体序列（如MOT17-07-FRCNN）的详细结果
   - 第三行是COMBINED（综合）的详细结果
   - 包含的指标类别：
     - HOTA系列：HOTA(0)、HOTA___5、HOTA___10等
     - DetA系列：检测精度
     - AssA系列：关联精度
     - DetRe系列：检测召回率
     - DetPr系列：检测精确率
     - AssRe系列：关联召回率
     - AssPr系列：关联精确率
     - LocA系列：定位精度
     - OWTA系列：最优权重跟踪精度
     - 以及其他详细指标

3. **summary.json**
   - 机器可读的JSON格式评估结果
   - 便于后续分析和处理

## 评估结果分析

### 结果解读

- **AIR-aircraft序列**：表现优秀，MOTA和IDF1均为100%，无误检和漏检
- **SAT-ship序列**：误检较多，需要调整参数以提高跟踪质量

### 改进方向

1. **参数调整**：根据评估结果调整跟踪参数
2. **提示词优化**：尝试更精确的提示词
3. **后处理**：添加目标过滤和轨迹平滑
4. **数据增强**：增加训练数据多样性
