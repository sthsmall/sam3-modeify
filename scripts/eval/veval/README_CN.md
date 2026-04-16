# SA-Co/VEval 数据集
**许可证** 每个域都有自己的许可证
* SA-Co/VEval - SA-V: CC-BY-NC 4.0
* SA-Co/VEval - YT-Temporal-1B: CC-BY-NC 4.0
* SA-Co/VEval - SmartGlasses: CC-by-4.0

**SA-Co/VEval** 是一个评估数据集，包含 3 个域，每个域都有 val 和 test 分割。
* SA-Co/VEval - SA-V: 视频来自 [SA-V 数据集](https://ai.meta.com/datasets/segment-anything-video/)
* SA-Co/VEval - YT-Temporal-1B: 视频来自 [YT-Temporal-1B](https://cove.thecvf.com/datasets/704)
* SA-Co/VEval - SmartGlasses: 来自 [Smart Glasses](https://huggingface.co/datasets/facebook/SACo-VEval/blob/main/media/saco_sg.tar.gz) 的第一人称视频

## 环境
安装 SA-Co/VEVal 所需的环境
```
pip install -e ".[veval]"
```
这将允许我们运行：
* `scripts/eval/veval/saco_yt1b_downloader.py` 为 SA-Co/VEval - YT-Temporal-1B 准备帧
* `examples/saco_veval_eval_example.ipynb` 运行离线评估器的示例
* `examples/saco_veval_vis_example.ipynb` 加载和可视化数据的示例

## 下载
### 预期的文件夹结构
完成本节中所有下载和预处理步骤后，预期的文件夹结构如下
```
data/
├── annotation/
│   ├── saco_veval_sav_test.json
│   ├── saco_veval_sav_val.json
│   ├── saco_veval_smartglasses_test.json
│   ├── saco_veval_smartglasses_val.json
│   ├── saco_veval_yt1b_test.json
│   ├── saco_veval_yt1b_val.json
└── media/
    ├── saco_sav
    │   └── JPEGImages_24fps
    ├── saco_sg
    │   └── JPEGImages_6fps
    └── saco_yt1b
        └── JPEGImages_6fps
```
### 下载即用数据
以下链接提供了即用数据，托管在 Roboflow 上，完成了下一节中概述的预处理步骤。

对于每个域：
- [SA-Co/VEval - SA-V](https://universe.roboflow.com/sa-co-veval/sa-v-test/)
- [SA-Co/VEval - YT-Temporal-1B](https://universe.roboflow.com/sa-co-veval/yt-temporal-1b-test/)
- [SA-Co/VEval - SmartGlasses](https://universe.roboflow.com/sa-co-veval/smartglasses-test/)

对于所有三个域：
- [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

关于 **SA-Co/VEval - YT-Temporal-1B** 的特别说明：
* **帧偏移警告！**
* Roboflow 上托管的即用数据是通过以下预处理步骤生成的。因此，YT-Temporal-1B 的帧偏移问题仍然存在：由于 YouTube 视频的性质，重新下载的视频可能与注释期间使用的视频不完全相同，这可能会影响评估数字的可重现性。

### 通过预处理步骤下载
#### 下载注释
GT 注释可在 Hugging Face 上获得：
* [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main)
    * SA-Co/VEval SA-V
        * 测试：`annotation/saco_veval_sav_test.json`
        * 验证：`annotation/saco_veval_sav_val.json`
    * SA-Co/VEval YT-Temporal-1B
        * 测试：`annotation/saco_veval_yt1b_test.json`
        * 验证：`annotation/saco_veval_yt1b_val.json`
    * SA-Co/VEval SmartGlasses
        * 测试：`annotation/saco_veval_smartglasses_test.json`
        * 验证：`annotation/saco_veval_smartglasses_val.json`

#### 下载视频或帧
##### SA-Co/VEval - SAV
按照 [SA-V 数据集](https://ai.meta.com/datasets/segment-anything-video/) 中的说明操作。只需要以下两个数据集：
* sav_test.tar
* sav_val.tar

解压后：
```
sav_test/
├── Annotations_6fps [忽略这是 SAM 2 注释]
├── JPEGImages_24fps
sav_val/
├── Annotations_6fps [忽略这是 SAM 2 注释]
└── JPEGImages_24fps
```
然后将两个 JPEGImages_24fps 合并在一起，以更好地匹配我们的注释 json 文件路径，例如
```
media/
    └── saco_sav
        └── JPEGImages_24fps [从上面的两个 JPEGImages_24fps 合并]
```
下载和合并文件夹的示例命令
```
cd ../data/media/saco_sav
wget -O sav_test.tar <从 SA-V 数据集页面下载 sav_test.tar 的链接>
wget -O sav_val.tar <从 SA-V 数据集页面下载 sav_val.tar 的链接>
tar -xf sav_test.tar
tar -xf sav_val.tar
mkdir JPEGImages_24fps
chmod -R u+w sav_test/
chmod -R u+w sav_val/
mv sav_test/JPEGImages_24fps/* JPEGImages_24fps/
mv sav_val/JPEGImages_24fps/* JPEGImages_24fps/
```

##### SA-Co/VEval - YT-Temporal-1B
下载 SA-Co/VEval - YT-Temporal-1B YouTube 视频需要两个文件。
* 从 [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main) 下载 `media/yt1b_start_end_time.json`，其中包含 YouTube 视频 ID 以及 SA-Co/VEval - YT-Temporal-1B 中使用的开始和结束时间。
* 准备 `cookies.txt` 文件。按照 yt-dlp 中的 [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) 和 [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) 说明准备 cookies_file。
    * 请参阅 yt-dlp 中关于 YouTube 账户被封禁风险的完整 **警告**！

然后运行 `scripts/eval/veval/saco_yt1b_downloader.py` 下载视频并准备帧，例如
```
python saco_yt1b_downloader.py \
--data_dir ../data/media/saco_yt1b \
--cookies_file ../data/media/saco_yt1b/cookies.txt \
--yt1b_start_end_time_file ../data/media/saco_yt1b/yt1b_start_end_time.json \
--yt1b_frame_prep_log_file ../data/media/saco_yt1b/yt1b_frame_prep.log
```
* data_dir: 下载 YouTube 视频并存储提取帧的目录
* cookies_file: 上面下载的 `cookies.txt`
* yt1b_start_end_time_file: 上面下载的 `yt1b_start_end_time.json`
* yt1b_frame_prep_log_file: 跟踪视频下载和帧提取状态的日志文件

然后运行 `scripts/eval/veval/saco_yt1b_annot_update.py` 根据视频可用性更新注释，例如
```
python saco_yt1b_annot_update.py \
--yt1b_media_dir ../data/media/saco_yt1b/JPEGImages_6fps \
--yt1b_input_annot_path ../data/annotation/saco_veval_yt1b_val.json \
--yt1b_output_annot_path ../data/annotation/saco_veval_yt1b_val_updated.json \
--yt1b_annot_update_log_path ../data/annotation/saco_veval_yt1b_val_updated.log
```

**注意**：
* 并非所有 YouTube 视频都可用，因为 YouTube 视频可能被删除或变为私有。脚本 `saco_yt1b_annot_update.py` 用于删除不可用视频的注释。
* **帧偏移警告！** 即使视频仍然可用，当从 YouTube 重新下载时，它们的规格（如 fps 和持续时间）可能与注释期间使用的不同。此外，有时 `ffmpeg` 似乎很难保证在不同环境中从同一视频中提取一致的帧。这可能会导致重新下载和重新提取的帧由于帧偏移而与我们的注释出现对齐问题。在 SA-Co/VEval - YT-Temporal-1B 上进行评估时请注意此警告。

##### SA-Co/VEval - SmartGlasses
前往 [SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main) 下载 `media/saco_sg.tar.gz`
```
cd ../data
hf download facebook/SACo-VEval media/saco_sg.tar.gz --repo-type dataset --local-dir .
cd ../data/media
tar -xzf saco_sg.tar.gz
```

## 注释格式
该格式类似于 [YTVIS](https://youtube-vos.org/dataset/vis/) 格式。

在注释 json 中，例如 `saco_veval_sav_test.json`，有 5 个字段：
* info:
    * 包含数据集信息的字典
    * 例如：{'version': 'v1', 'date': '2025-09-24', 'description': 'SA-Co/VEval SA-V Test'}
* videos
    * 当前注释 json 中使用的视频列表
    * 包含 {id, video_name, file_names, height, width, length}
* annotations
    * **正** masklet 及其相关信息的列表
    * 包含 {id, segmentations, bboxes, areas, iscrowd, video_id, height, width, category_id, noun_phrase}
        * video_id 应与上面的 `videos - id` 字段匹配
        * category_id 应与下面的 `categories - id` 字段匹配
        * segmentations 是 [RLE](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) 列表
* categories
    * **全局**使用的名词短语 id 映射，适用于所有 3 个域。
    * 包含 {id, name}
        * name 是名词短语
* video_np_pairs
    * 视频-np 对的列表，包括当前注释 json 中使用的**正**和**负**对
    * 包含 {id, video_id, category_id, noun_phrase, num_masklets}
        * video_id 应与上面的 `videos - id` 匹配
        * category_id 应与上面的 `categories - id` 匹配
        * 当 `num_masklets > 0` 时，它是一个正视频-np 对，呈现的 masklet 可以在 annotations 字段中找到
        * 当 `num_masklets = 0` 时，它是一个负视频-np 对，意味着根本没有 masklet 呈现
```
data {
    "info": info
    "videos": [video]
    "annotations": [annotation]
    "categories": [category]
    "video_np_pairs": [video_np_pair]
}
video {
    "id": int
    "video_name": str  # 例如 sav_000000
    "file_names": List[str]
    "height": int
    "width": width
    "length": length
}
annotation {
    "id": int
    "segmentations": List[RLE]
    "bboxes": List[List[int, int, int, int]]
    "areas": List[int]
    "iscrowd": int
    "video_id": str
    "height": int
    "width": int
    "category_id": int
    "noun_phrase": str
}
category {
    "id": int
    "name": str
}
video_np_pair {
    "id": int
    "video_id": str
    "category_id": int
    "noun_phrase": str
    "num_masklets" int
}
```
[sam3/examples/saco_veval_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_veval_vis_example.ipynb) 展示了数据格式和数据可视化的一些示例。

## 运行离线评估
已提供示例笔记本和评估脚本用于离线评估。
```
sam3/
├── examples/
│   └── saco_veval_eval_example.ipynb  # 此笔记本将加载评估结果或实时运行评估，并打印结果
└── sam3/eval/
    └── saco_veval_eval.py  # 此脚本将运行离线评估器
```
`saco_veval_eval.py` 支持两种模式：`one` 和 `all`。
* `one`：只会接受一对 gt 和 pred 文件进行评估
* `all`：将在所有 6 个 SACo/VEval 数据集上进行评估

使用示例
```
python saco_veval_eval.py one \
--gt_annot_file ../sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_gt.json \
--pred_file ../sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_pred.json \
--eval_res_file ../sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_eval_res.json
```
* `gt_annot_file`：GT 文件的位置
* `pred_file`：Pred 文件的位置
* `eval_res_file`：评估结果将写入的位置

```
python saco_veval_eval.py all \
--gt_annot_dir ../data/annotation \
--pred_dir ../data/pred \
--eval_res_dir ../data/pred
```
* `gt_annot_dir`：GT 文件的位置
* `pred_dir`：Pred 文件的位置
* `eval_res_dir`：评估结果将写入的位置