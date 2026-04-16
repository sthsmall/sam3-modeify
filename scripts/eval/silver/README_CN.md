# SA-Co/Silver 基准测试

SA-Co/Silver 是一个用于图像中可提示概念分割（PCS）的基准测试。该基准测试包含与文本标签（也称为名词短语，简称 NPs）配对的图像，每个图像都被详尽地注释了与标签匹配的所有对象实例的掩码。

SA-Co/Silver 包含 10 个子集，涵盖了各种领域，包括食物、艺术、机器人、驾驶等。与 SA-Co/Gold 不同，每个数据点只有一个地面实况，这意味着结果可能有更多的方差，并且倾向于低估模型性能，因为它们没有考虑每个查询可能的不同解释。

- BDD100k
- DROID
- Ego4D
- MyFoodRepo-273
- GeoDE
- iNaturalist-2017
- National Gallery of Art
- SA-V
- YT-Temporal-1B
- Fathomnet

本 README 包含有关如何下载和设置注释、图像数据以准备它们用于 SA-Co/Silver 评估的说明。

# 准备
## 下载注释

GT 注释可以从 [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Silver) 或 [Roboflow](https://universe.roboflow.com/sa-co-silver) 下载

## 下载图像和视频帧

### 图像数据集

#### GeoDE

评估所需的处理图像可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/geode/) 下载，或按照以下步骤准备处理图像。

1. 从 [GeoDE](https://geodiverse-data-collection.cs.princeton.edu/) 下载带有原始图像的数据集。
2. 将下载的文件提取到一个位置，例如 `<RAW_GEODE_IMAGES_FOLDER>`

3. 运行以下命令预处理图像并准备评估。处理后的图像将保存到 `<PROCESSED_GEODE_IMAGES_FOLDER>` 中指定的位置
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_geode_merged_test.json --raw_images_folder <RAW_GEODE_IMAGES_FOLDER> --processed_images_folder <PROCESSED_GEODE_IMAGES_FOLDER> --dataset_name geode
    ```

#### National Gallery of Art (NGA)

评估所需的处理图像可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/national-gallery-of-art/) 下载，或按照以下步骤准备处理图像。

1. 运行以下命令下载原始图像并预处理图像以准备评估。处理后的图像将保存到 `<PROCESSED_NGA_IMAGES_FOLDER>` 中指定的位置。
    ```
    python download_preprocess_nga.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_nga_art_merged_test.json --raw_images_folder <RAW_NGA_IMAGES_FOLDER> --processed_images_folder <PROCESSED_NGA_IMAGES_FOLDER>
    ```

#### Berkeley Driving Dataset (BDD) 100k

评估所需的处理图像可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/bdd100k-gwmh6/) 下载，或按照以下步骤准备处理图像。

1. 从 [BDD100k](http://bdd-data.berkeley.edu/download.html) 中的 `100K Images` 数据集下载带有原始图像的数据
2. 将下载的文件提取到一个位置，例如 `<RAW_BDD_IMAGES_FOLDER>`
3. 运行以下命令预处理图像并准备评估。处理后的图像将保存到 `<PROCESSED_BDD_IMAGES_FOLDER>` 中指定的位置
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_bdd100k_merged_test.json --raw_images_folder <RAW_BDD_IMAGES_FOLDER> --processed_images_folder <PROCESSED_BDD_IMAGES_FOLDER> --dataset_name bdd100k
    ```

#### Food Recognition Challenge 2022

1. 从 [网站](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022) 下载带有原始图像的数据。下载 `[Round 2] public_validation_set_2.0.tar.gz` 文件。
2. 将下载的文件提取到一个位置，例如 `<RAW_FOOD_IMAGES_FOLDER>`
3. 运行以下命令预处理图像并准备评估。处理后的图像将保存到 `<PROCESSED_FOOD_IMAGES_FOLDER>` 中指定的位置
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_food_rec_merged_test.json --raw_images_folder <RAW_FOOD_IMAGES_FOLDER> --processed_images_folder <PROCESSED_FOOD_IMAGES_FOLDER> --dataset_name food_rec
    ```

#### iNaturalist

评估所需的处理图像可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/inaturalist-2017/) 下载，或按照以下步骤准备处理图像。

1. 运行以下命令下载、在 `<RAW_INATURALIST_IMAGES_FOLDER>` 中提取图像并准备评估。处理后的图像将保存到 `<PROCESSED_INATURALIST_IMAGES_FOLDER>` 中指定的位置
    ```
    python download_inaturalist.py --raw_images_folder <RAW_INATURALIST_IMAGES_FOLDER> --processed_images_folder <PROCESSED_INATURALIST_IMAGES_FOLDER>
    ```

#### Fathomnet

评估所需的处理图像可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/fathomnet-kmz5d/) 下载，或按照以下步骤准备处理图像。

1. 安装 FathomNet API
    ```
    pip install fathomnet
    ```

2. 运行以下命令下载图像并准备评估。处理后的图像将保存到 `<PROCESSED_BDD_IMAGES_FOLDER>` 中指定的位置
    ```
    python download_fathomnet.py --processed_images_folder <PROCESSED_BFATHOMNET_IMAGES_FOLDER>
    ```

### 帧数据集

这些数据集对应于来自视频的单个帧的注释。文件 `CONFIG_FRAMES.yaml` 用于统一数据集的下载，如下所述。

在遵循其他数据集步骤之前，请更新 `CONFIG_FRAMES.yaml` 中的 `path_annotations` 路径，其中包含注释文件。

#### DROID

评估所需的处理帧可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/droid-cfual/) 下载，或按照以下步骤准备处理帧。

1. 安装 gsutil 包：
    ```bash
    pip install gsutil
    ```
2. 修改 `CONFIG_FRAMES.yaml` 中的 `droid_path` 变量。这是下载 DROID 数据的路径。
3. _可选_ 更新变量 `remove_downloaded_videos_droid` 以（不）在提取帧后删除视频。
4. 下载数据：
    ```bash
    python download_videos.py droid
    ```
5. 提取帧：
    ```bash
    python extract_frames.py droid
    ```

有关更多信息，请参阅 [DROID 网站](https://droid-dataset.github.io/droid/the-droid-dataset#-using-the-dataset)。

#### SA-V

评估所需的处理帧可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/sa-v) 下载，或按照以下步骤准备处理帧。

1. 按照 [Segment Anything 官方网站](https://ai.meta.com/datasets/segment-anything-video-downloads/) 中的说明获取下载链接（它们是动态链接）。
2. 更新 `CONFIG_FRAMES.yaml`：
    - 更新 `sav_path` 变量，帧将保存在此处。
    - 更新 `sav_videos_fps_6_download_path` 变量。复制粘贴您在步骤 1 中获得的列表中对应于 `videos_fps_6.tar` 的路径。
    - _可选_ 更新变量 `remove_downloaded_videos_sav` 以（不）在提取帧后删除视频。
3. 下载视频：
    ```bash
    python download_videos.py sav
    ```
4. 提取帧：
    ```
    python extract_frames.py sav
    ```

#### Ego4D

评估所需的处理帧可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/ego4d-w7fiu/) 下载，或按照以下步骤准备处理帧。

1. 在 [官方 Ego4D 网站](https://ego4d-data.org/docs/start-here/#license-agreement) 上查看并接受许可协议。
2. 配置 AWS 凭证。运行：
    ```bash
    pip install awscli
    aws configure
    ```
    并复制您在步骤 1 后收到的电子邮件中显示的值（您可以留空 "region name" 和 "output format"）。您可以验证变量是否正确设置：
    ```bash
    cat ~/.aws/credentials
    ```
3. 安装 Ego4D 库：
    ```bash
    pip install ego4d
    ```
4. 更新 `CONFIG_FRAMES.yaml`：
    - 按照您在步骤 2 后收到的电子邮件中的说明设置 AWS 凭证。修改以下变量：`aws_access_key_id` 和 `aws_secret_access_key`。
    - 更新 `ego4d_path` 变量，帧将保存在此处。
    - _可选_ 更新变量 `remove_downloaded_videos_ego4d` 以（不）在提取帧后删除视频。
5. 下载 Ego4D 数据集的 `clips` 子集：
    ```python
    python download_videos.py ego4d
    ```
6. 提取帧：
    ```
    python extract_frames.py ego4d
    ```

有关更多信息，请参阅 [官方 CLI](https://ego4d-data.org/docs/CLI/) 和 [关于视频的解释](https://ego4d-data.org/docs/data/videos/)。

#### YT1B

评估所需的处理帧可以从 [Roboflow](https://universe.roboflow.com/sa-co-silver/yt-temporal-1b/) 下载，或按照以下步骤准备处理帧。

1. 安装 yt-dlp 库：
    ```bash
    python3 -m pip install -U "yt-dlp[default]"
    ```
2. 按照 yt-dlp [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) 和 [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) 的说明创建 `cookies.txt` 文件。这是下载 YouTube 视频所必需的。然后，在 `CONFIG_FRAMES.yaml` 文件中的变量 `cookies_path` 中更新该文件的路径。
3. 更新 `CONFIG_FRAMES.yaml`：
    - 更新 `yt1b_path`，帧将保存在此处。
    - _可选_ 某些 YouTube 视频可能不再在 YouTube 上可用。在 `CONFIG_FRAMES.yaml` 中将 `update_annotation_yt1b` 设置为 `True` 以删除与这些视频对应的注释。请注意，评估结果将无法与其他报告的评估直接比较。
    - _可选_ 更新变量 `remove_downloaded_videos_yt1b` 以（不）在提取帧后删除视频。
4. 运行以下代码下载视频：
    ```
    python download_videos.py yt1b
    ```
5. 提取帧：
    ```
    python extract_frames.py yt1b
    ```

# 使用
## 可视化

- 可视化 GT 注释：[saco_gold_silver_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)

## 运行评估

SA-Co/Silver 的官方指标是 cgF1。有关详细信息，请参阅 SAM3 论文。
与 Gold 不同，silver 子集每个图像只有一个注释。因此，性能可能被低估，因为模型可能因选择有效但与人类标注者不同的解释而被错误地惩罚。

### 评估 SAM3

我们提供推理配置来重现 SAM3 的评估。
首先，请编辑文件 [eval_base.yaml](https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/eval_base.yaml)，其中包含您上面下载的图像和注释的路径。

有 10 个子集和同样多的配置要运行。
让我们以第一个子集为例。可以使用以下命令在本地运行推理（您可以调整 gpu 数量）：
```bash
python sam3/train/train.py -c configs/silver_image_evals/sam3_gold_image_bdd100k.yaml --use-cluster 0 --num-gpus 1
```
预测将被转储到 eval_base.yaml 中指定的文件夹中。

我们还提供对基于 SLURM 的集群推理的支持。编辑 eval_base.yaml 文件以反映您的 slurm 配置（分区、qos 等），然后运行

```bash
python sam3/train/train.py -c configs/silver_image_evals/sam3_gold_image_bdd100k.yaml --use-cluster 1
```

### 离线评估

如果您有 COCO 结果格式的预测（参见 [这里](https://cocodataset.org/#format-results)），那么我们提供脚本来轻松运行评估。

有关如何在所有子集上运行评估器并聚合结果的示例，请参阅以下笔记本：[saco_gold_silver_eval_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_eval_example.ipynb)

如果您有给定子集的预测文件，您可以使用独立脚本专门为该子集运行评估器。示例：
```bash
python scripts/eval/standalone_cgf1.py --pred_file /path/to/coco_predictions_segm.json --gt_files /path/to/annotations/silver_bdd100k_merged_test.json
```

# 结果
<table style="border-color:black;border-style:solid;border-width:1px;border-collapse:collapse;border-spacing:0;text-align:right" class="tg"><thead>
  <tr style="text-align:center">
    <th></th>
    <th colspan="3">平均</th>
    <th colspan="3">BDD100k</th>
    <th colspan="3">Droids</th>
    <th colspan="3">Ego4d</th>
    <th colspan="3">Food Rec</th>
    <th colspan="3">Geode</th>
    <th colspan="3">iNaturalist</th>
    <th colspan="3">Nga Art</th>
    <th colspan="3">SAV</th>
    <th colspan="3">YT1B</th>
    <th colspan="3">Fathomnet</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td>cgF1</td>
    <td>IL_MCC</td>
    <td>PmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
    <td>CGF1</td>
    <td>IL_MCC</td>
    <td>pmF1</td>
  </tr>
  <tr>
    <td>gDino-T</td> <td>3.09</td> <td>0.12</td> <td>19.75</td> <td>3.33</td> <td>0.17</td> <td>19.54</td> <td>4.26</td> <td>0.15</td> <td>28.38</td> <td>2.87</td> <td>0.1</td>
    <td>28.72</td> <td>0.69</td> <td>0.05</td> <td>13.88</td> <td>9.61</td> <td>0.24</td> <td>40.03</td> <td>0</td> <td>0</td> <td>1.97</td> <td>1.31</td> <td>0.09</td>
    <td>14.57</td> <td>5.18</td> <td>0.19</td> <td>27.25</td> <td>3.6</td> <td>0.16</td> <td>22.5</td> <td>0</td> <td>0</td> <td>0.64</td>
  </tr>
  <tr>
    <td>OWLv2*</td> <td>11.23</td> <td>0.32</td> <td>31.18</td> <td>14.97</td> <td>0.46</td> <td>32.34</td> <td>10.84</td> <td>0.36</td> <td>30.1</td> <td>7.36</td> <td>0.23</td>
    <td>31.99</td> <td>19.35</td> <td>0.44</td> <td>43.98</td> <td>27.04</td> <td>0.5</td> <td>54.07</td> <td>3.92</td> <td>0.14</td> <td>27.98</td> <td>8.05</td> <td>0.31</td>
    <td>25.98</td> <td>10.59</td> <td>0.32</td> <td>33.1</td> <td>10.15</td> <td>0.38</td> <td>26.7</td> <td>0.04</td> <td>0.01</td> <td>5.57</td>
  </tr>
  <tr>
    <td>OWLv2</td> <td>8.18</td> <td>0.23</td> <td>32.55</td> <td>8.5</td> <td>0.31</td> <td>27.79</td> <td>7.21</td> <td>0.25</td> <td>28.84</td> <td>5.64</td> <td>0.18</td>
    <td>31.35</td> <td>14.18</td> <td>0.32</td> <td>44.32</td> <td>13.04</td> <td>0.28</td> <td>46.58</td> <td>3.62</td> <td>0.1</td> <td>36.23</td> <td>7.22</td> <td>0.25</td>
    <td>28.88</td> <td>10.86</td> <td>0.32</td> <td>33.93</td> <td>11.7</td> <td>0.35</td> <td>33.43</td> <td>-0.14</td> <td>-0.01</td> <td>14.15</td>
  </tr>
  <tr>
    <td>LLMDet-L</td> <td>6.73</td> <td>0.17</td> <td>28.19</td> <td>1.69</td> <td>0.08</td> <td>19.97</td> <td>2.56</td> <td>0.1</td> <td>25.59</td> <td>2.39</td>
    <td>0.08</td> <td>29.92</td> <td>0.98</td> <td>0.06</td> <td>16.26</td> <td>20.82</td> <td>0.37</td> <td>56.26</td> <td>27.37</td> <td>0.46</td> <td>59.5</td>
    <td>2.17</td> <td>0.13</td> <td>16.68</td> <td>5.37</td> <td>0.19</td> <td>28.26</td> <td>3.73</td> <td>0.16</td> <td>23.32</td> <td>0.24</td> <td>0.04</td> <td>6.1</td>
  </tr>
  <tr>
    <td>Gemini 2.5</td> <td>9.67</td> <td>0.19</td> <td>45.51</td> <td>5.83</td> <td>0.19</td> <td>30.66</td> <td>5.61</td> <td>0.14</td> <td>40.07</td>
    <td>0.38</td> <td>0.01</td> <td>38.14</td> <td>10.92</td> <td>0.24</td> <td>45.52</td> <td>18.28</td> <td>0.26</td> <td>70.29</td> <td>26.57</td> <td>0.36</td>
    <td>73.81</td> <td>8.18</td> <td>0.2</td> <td>40.91</td> <td>9.48</td> <td>0.22</td> <td>43.1</td> <td>8.66</td> <td>0.23</td> <td>37.65</td> <td>2.8</td>
    <td>0.08</td> <td>34.99</td>
  </tr>
  <tr> <td>SAM3</td> <td>49.57</td> <td>0.76</td> <td>65.17</td> <td>46.61</td> <td>0.78</td> <td>60.13</td> <td>45.58</td> <td>0.76</td>
    <td>60.35</td> <td>38.64</td> <td>0.62</td> <td>62.56</td> <td>52.96</td> <td>0.79</td> <td>67.21</td> <td>70.07</td> <td>0.89</td>
    <td>78.73</td> <td>65.8</td> <td>0.82</td> <td>80.67</td> <td>38.06</td> <td>0.66</td> <td>57.62</td> <td>44.36</td> <td>0.67</td>
    <td>66.05</td> <td>42.07</td> <td>0.72</td> <td>58.36</td> <td>51.53</td> <td>0.86</td> <td>59.98</td>
  </tr>
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

来自 DROID 域的示例注释如下所示：

#### images

```
[
  {
    "id": 10000000,
    "file_name": "AUTOLab_failure_2023-07-07_Fri_Jul__7_18:50:36_2023_recordings_MP4_22008760/00002.jpg",
    "text_input": "the large wooden table",
    "width": 1280,
    "height": 720,
    "queried_category": "3",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  }
]
```

#### annotations

```
[
  {
    "area": 0.17324327256944444,
    "id": 1,
    "image_id": 10000000,
    "source": "created by SAM3",
    "bbox": [
      0.03750000149011612,
      0.5083333253860474,
      0.8382812738418579,
      0.49166667461395264
    ],
    "segmentation": {
      "counts": "[^R11]f03O0O100O2N100O1O100O100O100O100O1O100O100O100O100O100O1O10000O1O10000O1O100O10000O1O100O100O100O100O100O100O100O100O100O100O1O100O100O10000O100O100O100O101N100O1O011O0O1O101OO0010O100O1O100O2OO0100O100O100O100O100O10000O100O100O1O100O10000O1O100O100O100O10000O1O100O100O100O10000O1O10000O1O100O100O100O100O100O100O1O100O100O100O100O100O100O100O100O100O100O100O100O100O100O10000O100O100O1O100O10000O100O100O100O100O1O100O100O100O100O100O100O10O0100O100O2O000O1O10000O1O10000O100O100O100O1O100O100O100O100O100O100O100O100O100O100O100O100O1O100O100O100O10000O100O100O100O100O100O100O100O100O100O100O100O100O100O10000O100O100O100O100O100O100O1O10000O1O10000O100O1O100O100O100O100O100O100O100O100O10000O1O100O100O100O100O1O10000O10\MP@hNo?W1U@gNk?X1W@gNh?Y1Z@fNf?Y1\@fNc?[1^@dNb?[1`@dN_?]1b@bN^?]1e@aNZ?_1i@_NW?a1l@\NS?d1RAXNn>h1TAVNk>k1VATNj>k1XATNg>m1YASNg>m1YASNf>m1[ASNe>m1[ASNd>m1]ASNc>m1]ASNb>l1`ATN`>i1cAWN\>d1jA\NV>_1oAaNP>^1RBbNn=\1TBdNk=\1VBdNj=1`@dNGO02P2Z1h=L_AfNj0^1g=FmC;R<EoC;Q<DPD<o;DRD<n;DQD=n;DjAnN?^1g=DhAQO?\1h=DhAUO<W1l=EeAZO:R1P>F]ABa0h0Q>Hd@lNDV1e17S>k1iAWNW>i1hAXNW>j1gAWNY>i1fAXNY>j1eAWNZ>k1dAVN\>k1bAVN^>k1`AVN_>l1`ATN`>m1^ATNa>o1]AQNc>P2[AQNd>P2\APNd>Q2[AoMd>R2[AoMd>R2\AnMd>S2ZAnMe>S2[AmMe>T2YAmMf>T2YAmMg>T2WAmMh>U2VAlMj>U2TAlMl>U2PAnMo>U2j@PNV?e4O100O100O100O100O100O100O100O100O100O100O100O100O101N100O100O10O0100O100O100O100O100O100O1000000O1000000O100O100O1O1O1O100O100O1O100O100O100O100O100O100O100O100O100O1O100O100O100O100O100O10000O100O1O100O100O100O100O100O100OkK_B]Oa=7oBEP=4YCKg<1^CNa<1bCN^<OeC1[<LhC4W<KlC4S<KoC5Q<JPD6o;JRD6n;JSD5l;LTD4l;LTD4k;MUD3k;MUD4j;LWD2i;OWD1i;OWD1h;0XD0h;1WDOh;2XDOg;1ZDNe;3[DMe;3[DNc;3]DLd;4\DLc;5]DKb;7]DIc;7^DHa;9_DGa;9_DG`;:`DF`;;_DE`;<`DCa;=^DDa;=_DC`;>_DCa;>^DBb;[OUCiMW1n2c;YO[CeMn0V3g;TO^CeMf0[3k;POaCdM>b3Q<iNbCfM7f3V<dNeCeMKQ4`<YNgCfMAX4g<RNiCk2W<SMlCl2S<TMnCl2R<SMoCm2Q<RMQDm2n;TMRDl2n;SMTDl2k;UMUDk2k;UMVDj2i;VMXDj2h;VMXDj2g;VM[Di2e;VM\Dj2c;VM^Dj2b;TMaDk2^;PMhDP3X;aL`CjM`1e5o:\L^Ed3b:WLd[... 2375 chars omitted ...]
      "size": [
        720,
        1280
      ]
    },
    "category_id": 1,
    "iscrowd": 0
  }
]
```

### 数据统计

以下是 10 个注释域的统计数据。# Image-NPs 表示唯一图像-NP 对的总数，包括 "正" 和 "负" NPs。


| 域 | # Image-NPs | # Image-NP-Masks |
|--------------------------|--------------| ----------------|
| BDD100k | 5546 | 13210 |
| DROID | 9445 | 11098 |
| Ego4D | 12608 | 24049 |
| MyFoodRepo-273 | 20985 | 28347 |
| GeoDE | 14850 | 7570 |
| iNaturalist-2017 | 1439051 | 48899 |
| National Gallery of Art | 22294 | 18991 |
| SA-V | 18337 | 39683 |
| YT-Temporal-1B | 7816 | 12221 |
| Fathomnet | 287193 | 14174 |