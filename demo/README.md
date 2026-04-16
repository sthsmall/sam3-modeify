# SAM3 演示示例

本目录包含 SAM3 (Segment Anything Model 3) 的基本演示示例，按功能类型分为多个文件。

## 演示文件说明

### 1. 图像分割示例
- **demo_image_segmentation.py** - 使用文本提示进行图像分割
- **demo_interactive_segmentation.py** - 使用鼠标交互进行图像分割

### 2. 视频处理示例
- **demo_video_tracking.py** - 视频目标跟踪
- **demo_video_segmentation.py** - 视频分割

### 3. 文本提示示例
- **demo_text_prompt.py** - 文本提示分割示例

## 环境依赖

请按照项目根目录的 README.md 中的安装说明进行环境配置：

```bash
# 创建并激活环境
conda create -n sam3 python=3.12
conda activate sam3

# 安装 PyTorch
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装 SAM3 及其依赖
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 安装额外依赖
pip install -e ".[notebooks]"
```

## 模型准备

1. 首先在 Hugging Face 上请求 SAM3 模型的访问权限：[https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. 安装 Hugging Face CLI 并登录：
   ```bash
   pip install huggingface_hub
   hf auth login
   ```
3. 模型文件会在首次运行时自动下载。

## 使用方法

### 运行图像分割示例
```bash
python demo_image_segmentation.py
```

### 运行交互式分割示例
```bash
python demo_interactive_segmentation.py
```

### 运行视频跟踪示例
```bash
python demo_video_tracking.py
```

### 运行视频分割示例
```bash
python demo_video_segmentation.py
```

### 运行文本提示示例
```bash
python demo_text_prompt.py
```

## 注意事项

1. 确保已安装所有依赖包
2. 确保已在 Hugging Face 上获取模型访问权限
3. 对于视频示例，需要准备测试视频文件 `test_video.mp4`
4. 对于图像示例，需要准备测试图像文件 `test_image.jpg`
5. 交互式示例需要图形界面支持
6. 首次运行时会自动下载模型文件，可能需要一些时间

## 序列转视频工具

如果您的数据集是MOT格式的图像序列（如提供的 `D:\dataset\laboratory_data\laboratory_data`），可以使用以下工具将其转换为视频文件：

### 单个序列转换
```bash
python demo/utils/sequence_to_video.py <序列目录> <输出视频路径> --fps <帧率>
```

示例：
```bash
python demo/utils/sequence_to_video.py "D:\dataset\laboratory_data\laboratory_data\AIR-ship\test\MOT17-12-FRCNN" "output.mp4" --fps 30
```

### 批量转换
```bash
python demo/utils/batch_sequence_to_video.py <数据集根目录> <输出视频根目录> --fps <帧率>
```

示例：
```bash
python demo/utils/batch_sequence_to_video.py "D:\dataset\laboratory_data\laboratory_data" "D:\dataset\laboratory_data\videos" --fps 30
```

转换后的视频可以用于 `demo_video_tracking.py` 和 `demo_video_segmentation.py` 中的测试。