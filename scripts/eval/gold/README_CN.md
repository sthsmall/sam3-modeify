# SA-Co/Gold 基准测试

SA-Co/Gold 是一个用于图像中可提示概念分割（PCS）的基准测试。该基准测试包含与文本标签（也称为名词短语，NPs）配对的图像，每个图像都被详尽地注释了与标签匹配的所有对象实例的掩码。SA-Co/Gold 包含 7 个子集，每个子集针对不同的注释域：MetaCLIP  captioner NPs、SA-1B captioner NPs、Attributes、Crowded Scenes、Wiki-Common1K、Wiki-Food/Drink、Wiki-Sports Equipment。这些图像最初来自 MetaCLIP 和 SA-1B 数据集。

对于每个子集，注释由 3 个独立的人类标注者进行多重审查。图中的每一行显示来自其中一个域的图像和名词短语对，以及来自 3 个标注者的掩码。虚线边框表示覆盖多个实例的特殊组掩码，当分离成实例被认为过于困难时使用。标注者有时会在精确的掩码边界、实例数量以及短语是否存在方面存在分歧。拥有 3 个独立的注释允许我们测量人类在任务上的一致性，这作为模型性能的上限。


<p align="center">
  <img src="../../../assets/saco_gold_annotation.png?" style="width:80%;" />
</p>
# 准备

## 下载注释

GT 注释可以从 [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Gold) 或 [Roboflow](https://universe.roboflow.com/sa-co-gold) 下载

## 下载图像

评估数据集有两个图像来源：MetaCLIP 和 SA-1B。

1) MetaCLIP 图像在 7 个子集中的 6 个中被引用（MetaCLIP captioner NPs、Attributes、Crowded Scenes、Wiki-Common1K、Wiki-Food/Drink、Wiki-Sports Equipment），可以从 [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-metaclip-merged-a-release-test/) 下载。

2) SA-1B 图像在 7 个子集中的 1 个中被引用（SA-1B captioner NPs），可以从 [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-sa-1b-merged-a-release-test/) 下载。或者，它们可以从 [这里](https://ai.meta.com/datasets/segment-anything-downloads/) 下载。请从 `Download text file` 下可用的动态链接访问 `sa_co_gold.tar` 链接，以下载 SA-Co/Gold 中引用的 SA-1B 图像。

# 使用
## 可视化

- 可视化 GT 注释：[saco_gold_silver_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)
- 并排可视化 GT 注释和样本预测：[sam3_data_and_predictions_visualization.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_data_and_predictions_visualization.ipynb)


## 运行评估

SA-Co/Gold 的官方指标是 cgF1。有关详细信息，请参阅 SAM3 论文。
我们的评估器继承自官方 COCO 评估器，并有一些修改。请记住，在 Gold 子集中，每个数据点有三个注释。我们针对每个注释进行评估，并选择最有利的（oracle 设置）。它具有最小的依赖项（pycocotools、numpy 和 scipy），以帮助在其他项目中重用。在本节中，我们提供了几个运行 SAM3 或第三方模型评估的指针。

### 评估 SAM3

我们提供推理配置来重现 SAM3 的评估。
首先，请编辑文件 [eval_base.yaml](https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/eval_base.yaml)，其中包含您上面下载的图像和注释的路径。

有 7 个子集和同样多的配置要运行。
让我们以第一个子集为例。可以使用以下命令在本地运行推理（您可以调整 gpu 数量）：
```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 0 --num-gpus 1
```
预测将被转储到 eval_base.yaml 中指定的文件夹中。

我们还提供对基于 SLURM 的集群推理的支持。编辑 eval_base.yaml 文件以反映您的 slurm 配置（分区、qos 等），然后运行

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 1
```

我们在下面提供所有子集的命令
#### MetaCLIP captioner NPs

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 1
```
#### SA-1B captioner NPs

此子集参考 SA-1B 图像。对于其他 6 个子集，请参考 MetaCLIP 图像。

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml --use-cluster 1
```
#### Attributes

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_attributes.yaml --use-cluster 1
```
#### Crowded Scenes

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_crowded.yaml --use-cluster 1
```
#### Wiki-Common1K

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_wiki_common.yaml --use-cluster 1
```
#### Wiki-Food/Drink

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_fg_food.yaml --use-cluster 1
```

#### Wiki-Sports Equipment

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_fg_sports.yaml --use-cluster 1
```

### 离线评估

如果您有 COCO 结果格式的预测（参见 [这里](https://cocodataset.org/#format-results)），那么我们提供脚本来轻松运行评估。

有关如何在所有子集上运行评估器并聚合结果的示例，请参阅以下笔记本：[saco_gold_silver_eval_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_eval_example.ipynb)
或者，您可以运行 `python scripts/eval/gold/eval_sam3.py`

如果您有给定子集的预测文件，您可以使用独立脚本专门为该子集运行评估器。示例：
```bash
python scripts/eval/standalone_cgf1.py --pred_file /path/to/coco_predictions_segm.json --gt_files /path/to/annotations/gold_metaclip_merged_a_release_test.json  /path/to/annotations/gold_metaclip_merged_b_release_test.json  /path/to/annotations/gold_metaclip_merged_c_release_test.json
```


# 结果
这里我们收集了 SAM3 和一些基线的分割结果。请注意，不产生掩码的基线通过使用 SAM2 将框转换为掩码来评估
<table style="border-color:black;border-style:solid;border-width:1px;border-collapse:collapse;border-spacing:0;text-align:right" class="tg"><thead>
<tr><th style="text-align:center"></th><th style="text-align:center" colspan="3">平均</th><th style="text-align:center" colspan="3">Captioner metaclip</th><th style="text-align:center" colspan="3">Captioner sa1b</th>
<th style="text-align:center" colspan="3">Crowded</th><th style="text-align:center" colspan="3">FG food</th><th style="text-align:center" colspan="3">FG sport</th><th style="text-align:center" colspan="3">Attributes</th>
<th style="text-align:center" colspan="3">Wiki common</th></tr>
</thead>
<tbody>
<tr><td ></td><td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td>
<td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td>
<td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td><td >IL_MCC</td>
<td >positive_micro_F1</td><td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td>
<td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td>
<td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td><td >IL_MCC</td>
<td >positive_micro_F1</td></tr>
<tr><td >gDino-T</td><td >3.25</td><td >0.15</td><td >16.2</td>
<td >2.89</td><td >0.21</td><td >13.88</td><td >3.07</td>
<td >0.2</td><td >15.35</td><td >0.28</td><td >0.08</td>
<td >3.37</td><td >0.96</td><td >0.1</td><td >9.83</td>
<td >1.12</td><td >0.1</td><td >11.2</td><td >13.75</td>
<td >0.29</td><td >47.3</td><td >0.7</td><td >0.06</td>
<td >12.14</td></tr>
<tr><td >OWLv2*</td><td >24.59</td><td >0.57</td><td >42</td>
<td >17.69</td><td >0.52</td><td >34.27</td><td >13.32</td>
<td >0.5</td><td >26.83</td><td >15.8</td><td >0.51</td>
<td >30.74</td><td >31.96</td><td >0.65</td><td >49.35</td>
<td >36.01</td><td >0.64</td><td >56.19</td><td >35.61</td>
<td >0.63</td><td >56.23</td><td >21.73</td><td >0.54</td>
<td >40.25</td></tr>
<tr><td >OWLv2</td><td >17.27</td><td >0.46</td><td >36.8</td>
<td >12.21</td><td >0.39</td><td >31.33</td><td >9.76</td>
<td >0.45</td><td >21.65</td><td >8.87</td><td >0.36</td>
<td >24.77</td><td >24.36</td><td >0.51</td><td >47.85</td>
<td >24.44</td><td >0.52</td><td >46.97</td><td >25.85</td>
<td >0.54</td><td >48.22</td><td >15.4</td><td >0.42</td>
<td >36.64</td></tr>
<tr><td >LLMDet-L</td><td >6.5</td><td >0.21</td><td >27.3</td>
<td >4.49</td><td >0.23</td><td >19.36</td><td >5.32</td>
<td >0.23</td><td >22.81</td><td >2.42</td><td >0.18</td>
<td >13.74</td><td >5.5</td><td >0.19</td><td >29.12</td>
<td >4.39</td><td >0.17</td><td >25.34</td><td >22.17</td>
<td >0.39</td><td >57.13</td><td >1.18</td><td >0.05</td>
<td >23.3</td></tr>
<tr><td >APE</td><td >16.41</td><td >0.4</td><td >36.9</td>
<td >12.6</td><td >0.42</td><td >30.11</td><td >2.23</td>
<td >0.22</td><td >10.01</td><td >7.15</td><td >0.35</td>
<td >20.3</td><td >22.74</td><td >0.51</td><td >45.01</td>
<td >31.79</td><td >0.56</td><td >56.45</td><td >26.74</td>
<td >0.47</td><td >57.27</td><td >11.59</td><td >0.29</td>
<td >39.46</td></tr>
<tr><td >DINO-X</td><td >21.26</td><td >0.38</td><td >55.2</td>
<td >17.21</td><td >0.35</td><td >49.17</td><td >19.66</td>
<td >0.48</td><td >40.93</td><td >12.86</td><td >0.34</td>
<td >37.48</td><td >30.07</td><td >0.49</td><td >61.72</td>
<td >28.36</td><td >0.41</td><td >69.4</td><td >30.97</td>
<td >0.42</td><td >74.04</td><td >9.72</td><td >0.18</td>
<td >53.52</td></tr>
<tr><td >Gemini 2.5</td><td >13.03</td><td >0.29</td><td >46.1</td>
<td >9.9</td><td >0.29</td><td >33.79</td><td >13.1</td>
<td >0.41</td><td >32.1</td><td >8.15</td><td >0.27</td>
<td >30.34</td><td >19.63</td><td >0.33</td><td >59.52</td>
<td >15.07</td><td >0.28</td><td >53.5</td><td >18.84</td>
<td >0.3</td><td >63.14</td><td >6.5</td><td >0.13</td>
<td >50.32</td></tr>
<tr><td >SAM 3</td><td >54.06</td><td >0.82</td><td >66.11</td>
<td >47.26</td><td >0.81</td><td >58.58</td><td >53.69</td>
<td >0.86</td><td >62.55</td><td >61.08</td><td >0.9</td>
<td >67.73</td><td >53.41</td><td >0.79</td><td >67.28</td>
<td >65.52</td><td >0.89</td><td >73.75</td><td >54.93</td>
<td >0.76</td><td >72</td><td >42.53</td><td >0.7</td>
<td >60.85</td></tr>
</tbody></table>



# 注释格式

注释格式源自 [COCO 格式](https://cocodataset.org/#format-data)。值得注意的数据字段包括：

- `images`：`dict` 特征的 `list`，包含所有图像-NP 对的列表。每个条目与一个图像-NP 对相关，具有以下项目。
  - `id`：`int` 特征，图像-NP 对的唯一标识符
  - `text_input`：`string` 特征，图像-NP 对的名词短语
  - `file_name`：`string` 特征，相应数据文件夹中的相对图像路径。
  - `height`/`width`：图像的维度
  - `is_instance_exhaustive`：布尔值（0 或 1）。如果为 1，则所有实例都被正确注释。对于实例分割，我们只使用那些数据点。否则，可能存在缺失的实例或人群段（覆盖多个实例的段）
  - `is_pixel_exhaustive`：布尔值（0 或 1）。如果为 1，则所有掩码的并集覆盖与提示对应的所有像素。这比 instance_exhaustive 弱，因为它允许人群段。它可用于语义分割评估。

- `annotations`：`dict` 特征的 `list`，包含所有注释的列表，包括边界框、分割掩码、面积等。
  - `image_id`：`int` 特征，映射到图像中图像-np 对的标识符
  - `bbox`：浮点特征的 `list`，包含 [x,y,w,h] 格式的边界框，按图像维度归一化
  - `segmentation`：dict 特征，包含 RLE 格式的分割掩码
  - `category_id`：为了与 coco 格式兼容。将始终为 1 且未使用。
  - `is_crowd`：布尔值（0 或 1）。如果为 1，则该段与多个实例重叠（用于实例不可分离的情况，例如由于图像质量差）

- `categories`：`dict` 特征的 `list`，包含所有类别的列表。在这里，我们提供类别键以与 COCO 格式兼容，但在开放词汇检测中我们不使用它。相反，文本提示直接存储在每个图像中（图像中的 text_input）。请注意，在我们的设置中，唯一的图像（图像中的 id）实际上对应于（图像，文本提示）组合。


对于在图像中具有相应注释的 `id`（即作为 `annotations` 中的 `image_id` 存在），我们将其称为 "正" NP。对于在 `images` 中没有任何注释的 `id`（即它们不作为 `annotations` 中的 `image_id` 存在），我们将其称为 "负" NP。

来自 Wiki-Food/Drink 域的示例注释如下所示：

#### images

```
[
  {
    "id": 10000000,
    "file_name": "1/1001/metaclip_1_1001_c122868928880ae52b33fae1.jpeg",
    "text_input": "chili",
    "width": 600,
    "height": 600,
    "queried_category": "0",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  },
  {
    "id": 10000001,
    "file_name": "1/1001/metaclip_1_1001_c122868928880ae52b33fae1.jpeg",
    "text_input": "the fish ball",
    "width": 600,
    "height": 600,
    "queried_category": "2001",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  }
]
```

#### annotations

```
[
  {
    "id": 1,
    "image_id": 10000000,
    "source": "manual",
    "area": 0.002477777777777778,
    "bbox": [
      0.44333332777023315,
      0.0,
      0.10833333432674408,
      0.05833333358168602
    ],
    "segmentation": {
      "counts": "`kk42fb01O1O1O1O001O1O1O001O1O00001O1O001O001O0000000000O1001000O010O02O001N10001N0100000O10O1000O10O010O100O1O1O1O1O0000001O0O2O1N2N2Nobm4",
      "size": [
        600,
        600
      ]
    },
    "category_id": 1,
    "iscrowd": 0
  },
  {
    "id": 2,
    "image_id": 10000000,
    "source": "manual",
    "area": 0.001275,
    "bbox": [
      0.5116666555404663,
      0.5716666579246521,
      0.061666667461395264,
      0.036666665226221085
    ],
    "segmentation": {
      "counts": "aWd51db05M1O2N100O1O1O1O1O1O010O100O10O10O010O010O01O100O100O1O00100O1O100O1O2MZee4",
      "size": [
        600,
        600
      ]
    },
    "category_id": 1,
    "iscrowd": 0
  }
]
```

# 数据统计

以下是 7 个注释域的统计数据。# Image-NPs 表示唯一图像-NP 对的总数，包括 "正" 和 "负" NPs。


| 域 | 媒体 | # Image-NPs | # Image-NP-Masks |
|--------------------------|--------------|---------------| ----------------|
| MetaCLIP captioner NPs | MetaCLIP | 33393 | 20144 |
| SA-1B captioner NPs | SA-1B | 13258 | 30306 |
| Attributes | MetaCLIP | 9245 | 3663 |
| Crowded Scenes | MetaCLIP | 20687 | 50417 |
| Wiki-Common1K | MetaCLIP | 65502 | 6448 |
| Wiki-Food&Drink | MetaCLIP | 13951 | 9825 |
| Wiki-Sports Equipment | MetaCLIP | 12166 | 5075 |