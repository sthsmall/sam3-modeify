# SAM3 视频分割与跟踪示例

本目录包含使用 SAM3 (Segment Anything Model 3) 进行图像分割和视频跟踪的示例代码。

## 文件说明

### 模型文件
- `sam3.pt` - SAM3 模型权重文件

### 测试脚本

#### test1.py - 交互式图像分割
使用边界框提示对单张图像进行交互式分割。

**功能：**
- 加载图像并显示
- 用户通过鼠标拖拽绘制边界框选择对象
- 支持选择多个区域
- 实时显示分割结果

**使用方法：**
```bash
python test1.py
```

**操作说明：**
- 拖拽绘制矩形框
- 按 ENTER 确认选择
- 按 `d` 处理所有选中的框
- 按 `q` 退出

---

#### test2.py - 文本提示图像分割
使用文本提示对单张图像进行分割。

**功能：**
- 使用文本提示（如 "plane"）自动检测并分割对象
- 在原图上绘制分割结果
- 保存结果图像

**使用方法：**
```bash
python test2.py
```

**参数：**
- `text=["plane"]` - 文本提示词

---

#### test3.py - 视频逐帧分割
对视频进行逐帧分割处理。

**功能：**
- 逐帧读取视频
- 使用文本提示分割每帧
- 输出分割后的视频

**使用方法：**
```bash
python test3.py
```

**说明：**
- 使用 SAM3SemanticPredictor
- 每帧独立处理，不进行跨帧跟踪

---

#### test4.py - 边界框参考视频跟踪
使用初始帧的边界框作为参考，自动检测并跟踪所有相似对象。

**功能：**
- 在第一帧框选参考对象
- 基于视觉相似性自动检测所有相似对象
- 为每个检测到的对象分配唯一 ID
- 跨帧跟踪所有对象

**使用方法：**
```bash
python test4.py
```

**操作说明：**
- 在第一帧框选一个或多个参考对象
- 按 `d` 开始跟踪
- 按 `q` 退出

**关键参数：**
```python
score_threshold_detection=0.3  # 检测阈值（越低越敏感）
max_trk_keep_alive=60          # 跟踪保持帧数
init_trk_keep_alive=60          # 初始跟踪保持帧数
new_det_thresh=0.0             # 新检测阈值
masklet_confirmation_enable=False # 禁用 masklet 确认
```

**工作原理：**
1. 提取框选对象的视觉特征作为参考模板
2. 在后续帧中寻找视觉特征相似的对象
3. 自动为每个相似对象分配跟踪 ID
4. 跨帧持续跟踪所有检测到的对象

---

#### test5.py - 文本提示视频自动跟踪
使用文本提示自动检测并跟踪视频中的所有目标。

**功能：**
- 使用文本提示（如 "plane"）自动检测所有匹配对象
- 为每个对象分配唯一 ID
- 跨帧跟踪所有对象

**使用方法：**
```bash
python test5.py
```

**参数：**
- `text=["plane"]` - 文本提示词

---

## 跟踪器对比

| 特性 | test1 | test2 | test3 | test4 | test5 |
|------|--------|--------|--------|--------|--------|
| 图像分割 | ✓ | ✓ | ✗ | ✗ | ✗ |
| 视频处理 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 交互式框选 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 文本提示 | ✗ | ✓ | ✓ | ✗ | ✓ |
| 跨帧跟踪 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 自动检测相似对象 | ✗ | ✗ | ✗ | ✓ | ✓ |

## 常见问题

### Q1: 为什么跟踪过程中目标会消失？
**A:** 可能的原因：
1. 目标被遮挡或移出画面
2. `max_trk_keep_alive` 参数设置过小
3. 检测阈值 `score_threshold_detection` 过高

**解决方案：**
- 增加 `max_trk_keep_alive` 值（如 60-120）
- 降低 `score_threshold_detection` 值（如 0.2-0.3）

### Q2: test4 和 test5 的区别是什么？
**A:** 
- **test4**: 使用边界框作为视觉参考，基于视觉相似性检测对象
- **test5**: 使用文本提示进行语义检测

### Q3: 如何提高检测精度？
**A:** 
- 调整 `score_threshold_detection` 参数
- 提供更准确的边界框（test4）
- 使用更具体的文本提示（test5）

### Q4: 为什么需要 `compile=None`？
**A:** 
ultralytics 库的默认配置中 `compile=False`（布尔值）会导致错误，因为 `torch.compile` 只接受字符串或 `None`。显式设置 `compile=None` 可以禁用编译。

## 依赖环境

```bash
pip install ultralytics opencv-python numpy torch
```

## 模型下载

SAM3 模型文件 `sam3.pt` 需要从官方渠道下载并放置在当前目录。

## 输出文件

运行测试脚本后，会在当前目录生成以下输出文件：
- `segmentation_result.jpg` - 图像分割结果（test2）
- `video_segmentation_result.mp4` - 视频分割结果（test3）
- `video_tracking_result.mp4` - 视频跟踪结果（test4, test5）

## 技术支持

如有问题，请参考：
- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [SAM3 论文和代码](https://github.com/facebookresearch/segment-anything-3)
