# SAM 3: 用概念分割一切

Meta 超智能实验室

[Nicolas Carion](https://www.nicolascarion.com/)*,
[Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en)*,
[Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en)*,
[Shoubhik Debnath](https://scholar.google.com/citations?user=fb6FOfsAAAAJ&hl=en)*,
[Ronghang Hu](https://ronghanghu.com/)*,
[Didac Suris](https://www.didacsuris.com/)*,
[Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en)*,
[Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en)*,
[Haitham Khedr](https://hkhedr.com/)*, Andrew Huang,
[Jie Lei](https://jayleicn.github.io/),
[Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en),
[Baishan Guo](https://scholar.google.com/citations?user=BC5wDu8AAAAJ&hl=en),
Arpit Kalla, [Markus Marks](https://damaggu.github.io/),
[Joseph Greer](https://scholar.google.com/citations?user=guL96CkAAAAJ&hl=en),
Meng Wang, [Peize Sun](https://peizesun.github.io/),
[Roman Rädle](https://scholar.google.com/citations?user=Tpt57v0AAAAJ&hl=en),
[Triantafyllos Afouras](https://www.robots.ox.ac.uk/~afourast/),
[Effrosyni Mavroudi](https://scholar.google.com/citations?user=vYRzGGEAAAAJ&hl=en),
[Katherine Xu](https://k8xu.github.io/)°,
[Tsung-Han Wu](https://patrickthwu.com/)°,
[Yu Zhou](https://yu-bryan-zhou.github.io/)°,
[Liliane Momeni](https://scholar.google.com/citations?user=Lb-KgVYAAAAJ&hl=en)°,
[Rishi Hazra](https://rishihazra.github.io/)°,
[Shuangrui Ding](https://mark12ding.github.io/)°,
[Sagar Vaze](https://sgvaze.github.io/)°,
[Francois Porcher](https://scholar.google.com/citations?user=LgHZ8hUAAAAJ&hl=en)°,
[Feng Li](https://fengli-ust.github.io/)°,
[Siyuan Li](https://siyuanliii.github.io/)°,
[Aishwarya Kamath](https://ashkamath.github.io/)°,
[Ho Kei Cheng](https://hkchengrex.com/)°,
[Piotr Dollar](https://pdollar.github.io/)†,
[Nikhila Ravi](https://nikhilaravi.com/)†,
[Kate Saenko](https://ai.bu.edu/ksaenko.html)†,
[Pengchuan Zhang](https://pzzhang.github.io/pzzhang/)†,
[Christoph Feichtenhofer](https://feichtenhofer.github.io/)†

* 核心贡献者，° 实习生，† 项目负责人，组内顺序随机

[[`论文`](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)]
[[`项目`](https://ai.meta.com/sam3)]
[[`演示`](https://segment-anything.com/)]
[[`博客`](https://ai.meta.com/blog/segment-anything-model-3/)]
[[`BibTeX`](#引用-sam-3)]

![SAM 3 架构](assets/model_diagram.png?raw=true) SAM 3 是一个统一的基础模型，用于图像和视频中的可提示分割。它可以使用文本或视觉提示（如点、框和掩码）检测、分割和跟踪对象。与前身 [SAM 2](https://github.com/facebookresearch/sam2) 相比，SAM 3 引入了对由短文本短语或示例指定的开放词汇概念的所有实例进行 exhaustive 分割的能力。与之前的工作不同，SAM 3 可以处理大量的开放词汇提示。在我们新的 [SA-CO 基准测试](https://github.com/facebookresearch/sam3?tab=readme-ov-file#sa-co-dataset) 上，它达到了人类表现的 75-80%，该基准测试包含 27 万个独特概念，是现有基准测试的 50 倍以上。

这一突破由一个创新的数据引擎推动，该引擎已经自动注释了超过 400 万个独特概念，创建了迄今为止最大的高质量开放词汇分割数据集。此外，SAM 3 引入了一种新的模型架构，其特点是存在令牌，可提高密切相关文本提示之间的区分能力（例如，"穿白色衣服的球员"与"穿红色衣服的球员"），以及最小化任务干扰并随数据有效扩展的解耦检测器-跟踪器设计。

<p align="center">
  <img src="assets/dog.gif" width=380 />
  <img src="assets/player.gif" width=380 />
</p>

## 最新更新

**2026年3月27日 -- SAM 3.1 Object Multiplex 发布。它引入了一种共享内存方法，用于联合多目标跟踪，在不牺牲准确性的情况下显著提高速度。**

- 一套新的改进模型检查点（表示为 **SAM 3.1**）已在 [Hugging Face](https://huggingface.co/facebook/sam3.1) 上发布。有关完整详细信息，请参阅 [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md)。
  * 要使用新的 SAM 3.1 检查点，您需要此存储库中的最新模型代码。如果您已安装此存储库的早期版本，请从该存储库中提取最新代码（使用 `git pull`），然后按照下面的 [安装](#安装) 重新安装存储库。

## 安装

### 先决条件

- Python 3.12 或更高版本
- PyTorch 2.7 或更高版本
- 兼容 CUDA 的 GPU，CUDA 12.6 或更高版本

1. **创建一个新的 Conda 环境：**

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

2. **安装带有 CUDA 支持的 PyTorch：**

```bash
pip install torch==2.10.0 torchvision --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128
```

3. **克隆存储库并安装包：**

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

4. **安装示例笔记本或开发的其他依赖项：**

```bash
# 用于运行示例笔记本
pip install -e ".[notebooks]"

# 用于开发
pip install -e ".[train,dev]"
```

5. **用于更快推理的可选依赖项**
```bash
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

## 入门

⚠️ 在使用 SAM 3 之前，请在 SAM 3 Hugging Face [存储库](https://huggingface.co/facebook/sam3) 上请求访问检查点。获得批准后，您需要进行身份验证才能下载检查点。您可以通过运行以下 [步骤](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) 来完成此操作（例如，生成访问令牌后运行 `hf auth login`。）

### 基本用法

```python
import torch
#################################### 用于图像 ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# 加载模型
model = build_sam3_image_model()
processor = Sam3Processor(model)
# 加载图像
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# 用文本提示模型
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# 获取掩码、边界框和分数
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### 用于视频 ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # JPEG 文件夹或 MP4 视频文件
# 开始会话
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # 任意帧索引
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

## 示例

`examples` 目录包含演示如何使用各种类型提示的 SAM3 的笔记本：

- [`sam3_image_predictor_example.ipynb`](examples/sam3_image_predictor_example.ipynb)
  ：演示如何在图像上用文本和视觉框提示 SAM 3。
- [`sam3_video_predictor_example.ipynb`](examples/sam3_video_predictor_example.ipynb)
  ：演示如何在视频上用文本提示 SAM 3，并使用点进行进一步的交互式细化。
- [`sam3_image_batched_inference.ipynb`](examples/sam3_image_batched_inference.ipynb)
  ：演示如何在图像上运行 SAM 3 的批处理推理。
- [`sam3_agent.ipynb`](examples/sam3_agent.ipynb)：演示使用 SAM
  3 Agent 在图像上分割复杂文本提示。
- [`saco_gold_silver_vis_example.ipynb`](examples/saco_gold_silver_vis_example.ipynb)
  ：显示 SA-Co 图像评估集的几个示例。
- [`saco_veval_vis_example.ipynb`](examples/saco_veval_vis_example.ipynb)：
  显示 SA-Co 视频评估集的几个示例。

示例目录中还有其他笔记本，演示如何将 SAM 3 用于图像和视频中的交互式实例分割（SAM 1/2 任务），或作为 MLLM 的工具，以及如何在 SA-Co 数据集上运行评估。

要运行 Jupyter 笔记本示例：

```bash
# 确保您已安装笔记本依赖项
pip install -e ".[notebooks]"

# 启动 Jupyter 笔记本
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## 模型

SAM 3 由共享视觉编码器的检测器和跟踪器组成。它有 848M 参数。检测器是一个基于 DETR 的模型，以文本、几何和图像示例为条件。跟踪器继承了 SAM 2 转换器编码器-解码器架构，支持视频分割和交互式细化。

## 图像结果

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="3" style="border-right: 2px solid #ddd; padding: 12px 20px">模型</th>
      <th colspan="3" style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">实例分割</th>
      <th colspan="5" style="text-align: center; padding: 12px 20px">框检测</th>
    </tr>
    <tr>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">SA-Co/Gold</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">COCO</th>
      <th style="text-align: center; padding: 12px 20px">SA-Co/Gold</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP<sub>o</sub>
</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">人类</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">72.8</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">74.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">OWLv2*</td>
      <td style="text-align: center; padding: 10px 20px; color: #999">29.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px; color: #999">43.4</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">24.6</td>
      <td style="text-align: center; padding: 10px 20px; color: #999">30.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px; color: #999">45.5</td>
      <td style="text-align: center; padding: 10px 20px">46.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">23.9</td>
      <td style="text-align: center; padding: 10px 20px">24.5</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">DINO-X</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">38.5</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">21.3</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">52.4</td>
      <td style="text-align: center; padding: 10px 20px">56.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">22.5</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Gemini 2.5</td>
      <td style="text-align: center; padding: 10px 20px">13.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">13.0</td>
      <td style="text-align: center; padding: 10px 20px">16.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">14.4</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">37.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">48.5</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">54.1</td>
      <td style="text-align: center; padding: 10px 20px">40.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">53.6</td>
      <td style="text-align: center; padding: 10px 20px">56.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">55.7</td>
      <td style="text-align: center; padding: 10px 20px">55.7</td>
    </tr>
  </tbody>
</table>

<p style="text-align: center; margin-top: 10px; font-size: 0.9em; color: #ddd;">* 部分在 LVIS 上训练，AP<sub>o</sub> 指 COCO-O 准确性</p>

</div>

## 视频结果

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 2px solid #ddd; padding: 12px 20px">模型</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SA-V 测试</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">YT-Temporal-1B 测试</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SmartGlasses 测试</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVVIS 测试</th>
      <th style="text-align: center; padding: 12px 20px">BURST 测试</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">mAP</th>
      <th style="text-align: center; padding: 12px 20px">HOTA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">人类</td>
      <td style="text-align: center; padding: 10px 20px">53.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">70.5</td>
      <td style="text-align: center; padding: 10px 20px">71.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">78.4</td>
      <td style="text-align: center; padding: 10px 20px">58.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">72.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">30.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">58.0</td>
      <td style="text-align: center; padding: 10px 20px">50.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">69.9</td>
      <td style="text-align: center; padding: 10px 20px">36.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">63.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">36.3</td>
      <td style="text-align: center; padding: 10px 20px">44.5</td>
    </tr>
  </tbody>
</table>
</div>

## SA-Co 数据集

我们发布了 2 个图像基准测试，[SA-Co/Gold](scripts/eval/gold/README.md) 和
[SA-Co/Silver](scripts/eval/silver/README.md)，以及一个视频基准测试
[SA-Co/VEval](scripts/eval/veval/README.md)。这些数据集包含带有注释名词短语的图像（或视频）。每个图像/视频和名词短语对都用实例掩码和匹配该短语的每个对象的唯一 ID 进行注释。没有匹配对象的短语（负提示）没有掩码，在图中以红色字体显示。有关如何下载和在数据集上运行评估的更多详细信息，请参阅链接的 README。

* HuggingFace 主机：[SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold)、[SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver) 和 [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)
* Roboflow 主机：[SA-Co/Gold](https://universe.roboflow.com/sa-co-gold)、[SA-Co/Silver](https://universe.roboflow.com/sa-co-silver) 和 [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

![SA-Co 数据集](assets/sa_co_dataset.jpg?raw=true)

## 开发

要设置开发环境：

```bash
pip install -e ".[dev,train]"
```

要格式化代码：

```bash
ufmt format .
```

## 贡献

请参阅 [贡献](CONTRIBUTING.md) 和
[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

本项目根据 SAM 许可证授权 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 致谢

我们要感谢以下人员对 SAM 3 项目的贡献：Alex He、Alexander Kirillov、
Alyssa Newcomb、Ana Paula Kirschner Mofarrej、Andrea Madotto、Andrew Westbury、Ashley Gabriel、Azita Shokpour、
Ben Samples、Bernie Huang、Carleigh Wood、Ching-Feng Yeh、Christian Puhrsch、Claudette Ward、Daniel Bolya、
Daniel Li、Facundo Figueroa、Fazila Vhora、George Orlin、Hanzi Mao、Helen Klein、Hu Xu、Ida Cheng、Jake Kinney、
Jiale Zhi、Jo Sampaio、Joel Schlosser、Justin Johnson、Kai Brown、Karen Bergan、Karla Martucci、Kenny Lehmann、
Maddie Mintz、Mallika Malhotra、Matt Ward、Michelle Chan、Michelle Restrepo、Miranda Hartley、Muhammad Maaz、
Nisha Deo、Peter Park、Phillip Thomas、Raghu Nayani、Rene Martinez Doehner、Robbie Adkins、Ross Girshik、Sasha
Mitts、Shashank Jain、Spencer Whitehead、Ty Toledano、Valentin Gabeur、Vincent Cho、Vivian Lee、William Ngan、
Xuehai He、Yael Yungster、Ziqi Pang、Ziyi Dou、Zoe Quake。

## 引用 SAM 3

如果您在研究中使用 SAM 3 或 SA-Co 数据集，请使用以下 BibTeX 条目。

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```