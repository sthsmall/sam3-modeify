# 训练

此存储库支持在多节点设置或本地执行中在自定义数据集上微调 SAM3 模型。训练脚本位于 `sam3/train.py`，并使用 Hydra 配置管理来处理复杂的训练设置。


## 安装

```bash
cd sam3
pip install -e ".[train]"
```

### 训练脚本使用

主要训练脚本位于 `sam3/train.py`。它使用 Hydra 配置管理来处理复杂的训练设置。

#### 基本用法

```bash
# 示例：在 Roboflow 数据集上训练
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
# 示例：在 ODinW13 数据集上训练
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```
按照 [`Roboflow 100-VL`](https://github.com/roboflow/rf100-vl/) 下载 roboflow 100-vl 数据集。按照 [`GLIP`](https://github.com/microsoft/GLIP) 下载 ODinW 数据集。数据文件夹应按以下方式组织，并将您的 roboflow_vl_100_root 和 odinw_data_root 放在作业配置中。
```
roboflow_vl_100_root:
  13-lkc01
    train
    valid
    test
  2024-frc
  actions
  ...
odinw_data_root:
  AerialMaritimeDrone
    large
      train
      valid
      test
  Aquarium
  ...
```

#### 命令行参数

训练脚本支持多个命令行参数：

```bash
python sam3/train/train.py \
    -c CONFIG_NAME \
    [--use-cluster 0|1] \
    [--partition PARTITION_NAME] \
    [--account ACCOUNT_NAME] \
    [--qos QOS_NAME] \
    [--num-gpus NUM_GPUS] \
    [--num-nodes NUM_NODES]
```

**参数：**
- `-c, --config`：**必需。** 配置文件的路径（例如，`sam3/train/configs/roboflow_v100_full_ft_100_images.yaml`）
- `--use-cluster`：是否在集群上启动（0：本地，1：集群）。默认值：使用配置设置
- `--partition`：用于集群执行的 SLURM 分区名称
- `--account`：用于集群执行的 SLURM 账户名称
- `--qos`：SLURM QOS（服务质量）设置
- `--num-gpus`：每个节点的 GPU 数量。默认值：使用配置设置
- `--num-nodes`：分布式训练的节点数量。默认值：使用配置设置

#### 本地训练示例

```bash
# 单 GPU 训练
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# 单节点多 GPU 训练
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4

# 强制本地执行，即使配置指定了 GPU
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0
```

#### 集群训练示例

```bash
# 使用配置中的默认设置进行基本集群训练
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1

# 使用特定 SLURM 设置进行集群训练
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 1 \
    --partition gpu_partition \
    --account my_account \
    --qos high_priority \
    --num-gpus 8 \
    --num-nodes 2
```

### 配置文件

训练配置存储在 `sam3/train/configs/` 中。配置文件使用 Hydra 的 YAML 格式，支持：

- **数据集配置**：数据路径、转换和加载参数
- **模型配置**：架构设置、检查点路径和模型参数
- **训练配置**：批量大小、学习率、优化设置
- **启动器配置**：分布式训练和集群设置
- **日志配置**：TensorBoard、实验跟踪和输出目录

#### 关键配置部分

```yaml
# 数据集和检查点的路径
paths:
  bpe_path: /path/to/bpe/file
  dataset_root: /path/to/dataset
  experiment_log_dir: /path/to/logs

# 本地/集群执行的启动器设置
launcher:
  num_nodes: 1
  gpus_per_node: 2
  experiment_log_dir: ${paths.experiment_log_dir}

# 集群执行设置
submitit:
  use_cluster: True
  timeout_hour: 72
  cpus_per_task: 10
  partition: null
  account: null
```

### 监控训练

训练脚本自动设置日志记录并将输出保存到实验目录：

```bash
# 日志保存到配置中指定的 experiment_log_dir
experiment_log_dir/
├── config.yaml              # 原始配置
├── config_resolved.yaml     # 所有变量展开的解析配置
├── checkpoints/             # 模型检查点（如果 skip_checkpointing=False）
├── tensorboard/             # TensorBoard 日志
├── logs/                    # 文本日志
└── submitit_logs/           # 集群作业日志（如果使用集群）
```

您可以使用 TensorBoard 监控训练进度：

```bash
tensorboard --logdir /path/to/experiment_log_dir/tensorboard
```

### 数据集扫描的作业数组

Roboflow 和 ODinW 配置支持作业数组，用于在不同数据集上训练多个模型：

此功能通过以下方式特别启用：
```yaml
submitit:
  job_array:
    num_tasks: 100
    task_index: 0
```

配置包含 100 个 Roboflow 超类别的完整列表，`submitit.job_array.task_index` 根据数组作业索引自动选择要使用的数据集。

```bash
# 提交作业数组以在不同的 Roboflow 数据集上训练
# 作业数组索引从 all_roboflow_supercategories 中选择哪个数据集
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 1
```

### 重现 ODinW13 10-shot 结果
运行以下作业将在 ODinW13 seed 300 上获得结果，请参阅配置文件中的 `odinw_train.train_file: fewshot_train_shot10_seed300`。
```bash
# 示例：在 ODinW13 数据集上训练
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```
将 `odinw_train.train_file` 更改为 `fewshot_train_shot10_seed30` 和 `fewshot_train_shot10_seed3` 以获得其他两个种子的结果。最终结果是从三个种子聚合的。请注意，少数作业可能在训练期间发散，在这种情况下，我们只需使用发散前最后一个检查点的结果。


### 评估脚本使用
通过与训练配置类似的设置，训练脚本 `sam3/train.py` 也可以用于评估，当在作业配置中设置 `trainer.mode = val` 时。运行以下作业将在 RF100-VL 和 ODinW13 数据集上获得零样本结果。
```bash
# 示例：在 Roboflow 数据集上评估
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml
# 示例：在 ODinW13 数据集上评估
python sam3/train/train.py -c configs/odinw13/odinw_text_only.yaml
```