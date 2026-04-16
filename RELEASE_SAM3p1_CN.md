# 发布说明

## SAM 3.1 — 2026年3月27日

SAM 3.1 引入了 **Object Multiplex**，这是一种共享内存方法，用于联合多目标跟踪，在不牺牲准确性的情况下显著提高速度。此版本还包括新的模型检查点和优化的推理。

### Object Multiplex

SAM 3 的视频管道独立处理每个跟踪对象，这与对象数量成线性比例。Object Multiplex 将对象分组到固定容量的桶中并联合处理它们，大幅减少冗余计算。有关技术详细信息，请参阅 [SAM 3 论文](https://arxiv.org/abs/2511.16719) 的附录 H（Object Multiplex）。

<p align="center">
  <img src="assets/sam3.1_diagram.png" width="720" />
</p>

#### 主要改进
- **~7倍速度提升**：在单个 H100 GPU 上处理 128 个对象时，与 2025 年 11 月发布的 SAM 3 相比
- 推理优化，显著提高多目标跟踪效率：
  - 减少检测-跟踪器关联和其他启发式方法中的 CPU-GPU 同步
  - 增强 `torch.compile` 支持，改进操作融合
  - 批处理后处理和视觉编码器，提高 GPU 利用率
- 在 SA-Co/VEval 视频基准测试中结果喜忧参半，在 YT-Temporal-1B 上有显著改进（+2.1 cgF1）
- 在 7 个基准测试中的 6 个上改进了 VOS 性能，包括在具有挑战性的 MOSEv2 上提高了 +2.0

#### 推理效率

<p align="center">
  <img src="assets/sam3.1_efficiency.png" width="720" />
</p>

#### 带有文本提示的视频 PCS

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="3" style="border-right: 2px solid #ddd; padding: 12px 16px">模型</th>
      <th colspan="6" style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">SA-Co/VEval 基准测试测试集</th>
      <th colspan="4" style="text-align: center; padding: 10px 16px">公共基准测试</th>
    </tr>
    <tr>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">SA-V</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">YT-Temporal-1B</th>
      <th colspan="2" style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">SmartGlasses</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">LVVIS</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">BURST</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">YTVIS21</th>
      <th style="text-align: center; padding: 10px 16px">OVIS</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 10px 16px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">pHOTA</th>
      <th style="text-align: center; padding: 10px 16px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">pHOTA</th>
      <th style="text-align: center; padding: 10px 16px">cgF1</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">pHOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">test mAP</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">test HOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">val mAP</th>
      <th style="text-align: center; padding: 10px 16px">val mAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3</td>
      <td style="text-align: center; padding: 10px 16px">30.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">58.0</td>
      <td style="text-align: center; padding: 10px 16px">50.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">69.9</td>
      <td style="text-align: center; padding: 10px 16px">36.4</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">63.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">36.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">44.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">57.4</td>
      <td style="text-align: center; padding: 10px 16px">60.5</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3.1</td>
      <td style="text-align: center; padding: 10px 16px">30.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">58.7</td>
      <td style="text-align: center; padding: 10px 16px">52.9</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">70.7</td>
      <td style="text-align: center; padding: 10px 16px">36.3</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">64.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">34.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">43.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">56.6</td>
      <td style="text-align: center; padding: 10px 16px">61.5</td>
    </tr>
  </tbody>
</table>

</div>

#### 视频对象分割 (VOS)

<div align="center">
<table style="min-width: 60%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 2px solid #ddd; padding: 12px 16px">模型</th>
      <th colspan="5" style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">J&amp;F</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">G</th>
      <th style="text-align: center; padding: 10px 16px">J&amp;Ḟ</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 10px 16px">MOSEv1 val</th>
      <th style="text-align: center; padding: 10px 16px">DAVIS17 val</th>
      <th style="text-align: center; padding: 10px 16px">LVOSv2 val</th>
      <th style="text-align: center; padding: 10px 16px">SA-V val</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">SA-V test</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">YTVOS19 val</th>
      <th style="text-align: center; padding: 10px 16px">MOSEv2 val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3</td>
      <td style="text-align: center; padding: 10px 16px">78.4</td>
      <td style="text-align: center; padding: 10px 16px">92.2</td>
      <td style="text-align: center; padding: 10px 16px">88.5</td>
      <td style="text-align: center; padding: 10px 16px">83.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">84.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">89.7</td>
      <td style="text-align: center; padding: 10px 16px">60.3</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3.1</td>
      <td style="text-align: center; padding: 10px 16px">79.6</td>
      <td style="text-align: center; padding: 10px 16px">92.7</td>
      <td style="text-align: center; padding: 10px 16px">89.2</td>
      <td style="text-align: center; padding: 10px 16px">83.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">85.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">89.3</td>
      <td style="text-align: center; padding: 10px 16px">62.3</td>
    </tr>
  </tbody>
</table>
</div>

### 新的检查点

SAM 3.1 检查点可在 [Hugging Face 仓库](https://huggingface.co/facebook/sam3.1) 上获取。有关下载和身份验证说明，请参阅 [入门指南](README.md#getting-started)。

### 笔记本

- [`sam3.1_video_predictor_example.ipynb`](examples/sam3.1_video_predictor_example.ipynb)：演示如何使用带有 Object Multiplex 的 SAM 3.1 进行视频分割和使用文本和点提示的密集跟踪。

### 贡献者

[Arpit Kalla](https://github.com/arpitkalla), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Christian Puhrsch](https://github.com/cpuhrsch), [Ho Kei Cheng](https://hkchengrex.com/), [Joseph Greer](https://scholar.google.com/citations?user=guL96CkAAAAJ&hl=en), [Meng Wang](https://github.com/mengwa41), [Miran Heo](https://sites.google.com/view/miranheo), [Pengchuan Zhang](https://pzzhang.github.io/pzzhang/), [Roman Rädle](https://scholar.google.com/citations?user=Tpt57v0AAAAJ&hl=en), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en)