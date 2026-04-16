## MOT 评估指标计算方式速查表

### 一、基础统计量（所有指标的基础）

| 符号 | 含义 | 说明 |
|------|------|------|
| **TP** | True Positives | 正确匹配的检测框数 |
| **FP** | False Positives | 虚警（预测了但没有GT） |
| **FN** | False Negatives | 漏检（GT存在但没预测） |
| **IDS** | ID Switches | 同一GT轨迹被分配不同预测ID的次数 |
| **FM** | Fragmentations | 轨迹中断次数 |
| **GT** | Ground Truth | 真实目标框总数 |

---

### 二、核心指标计算公式

#### 1. **MOTA** (Multiple Object Tracking Accuracy)
```
MOTA = 1 - (FN + FP + IDS) / GT
```
- 范围：[-∞, 1]，越大越好
- 惩罚：漏检、虚警、ID切换

#### 2. **MOTP** (Multiple Object Tracking Precision)
```
MOTP = Σ(IoU_match) / Σ(matches)
```
- 范围：[0, 1]，越大越好
- 含义：所有匹配框的平均IoU

#### 3. **IDF1** (Identification F1 Score)
```
IDTP = 预测轨迹与GT轨迹匹配正确的帧数
IDFP = 预测轨迹没有对应GT的帧数
IDFN = GT轨迹没有对应预测的帧数

IDF1 = 2 × IDTP / (2×IDTP + IDFP + IDFN)
```
- 范围：[0, 1]，越大越好
- 含义：身份保持能力

#### 4. **HOTA** (Higher Order Tracking Accuracy)
```
DetA = TP / (TP + FN + FP)     # 检测精度
AssA = Σ(c(tp) / (c(tp) + c(fp) + c(fn))) / |TP|  # 关联精度

HOTA = √(DetA × AssA)
```
- 范围：[0, 1]，越大越好
- 含义：平衡检测和关联的综合指标

---

### 三、轨迹完整性指标

| 指标 | 计算方式 | 范围 |
|------|----------|------|
| **MT** (Mostly Tracked) | 匹配率 > 80% 的轨迹数 | 整数，越大越好 |
| **PT** (Partially Tracked) | 匹配率 20%-80% 的轨迹数 | 整数，中等 |
| **ML** (Mostly Lost) | 匹配率 < 20% 的轨迹数 | 整数，越小越好 |

**匹配率** = 轨迹匹配上的帧数 / 轨迹总帧数

---

### 四、错误统计指标

| 指标 | 含义 | 计算 |
|------|------|------|
| **FP** | 总虚警数 | Σ(未匹配的预测框) |
| **FN** | 总漏检数 | Σ(未匹配的GT框) |
| **IDS** | ID切换总数 | Σ(预测ID对应的GT_ID变化次数) |
| **FM** | 轨迹中断总数 | Σ(轨迹重新出现的次数) |

---

### 五、计算步骤总结

```
步骤1：帧级匹配
   输入：每帧的GT框和预测框
   方法：匈牙利算法 + IoU阈值(0.5)
   输出：匹配对、FP、FN

步骤2：轨迹级ID分配
   输入：所有帧的匹配结果
   方法：基于覆盖率的最大二分图匹配
   输出：每个预测轨迹对应的GT ID

步骤3：统计基础量
   TP = 匹配上的帧数
   FP = 未匹配的预测帧数
   FN = 未匹配的GT帧数
   IDS = 预测ID对应的GT_ID变化次数

步骤4：计算指标
   MOTA = 1 - (FN+FP+IDS)/GT
   IDF1 = 2×IDTP/(2×IDTP+IDFP+IDFN)
   HOTA = √(DetA × AssA)
```

---

### 六、快速计算示例

```python
# 假设数据
GT = 100      # 总GT框数
FN = 10       # 漏检10个
FP = 5        # 虚警5个
IDS = 2       # ID切换2次

# MOTA
MOTA = 1 - (10+5+2)/100 = 0.83

# IDF1 (假设)
IDTP = 85     # 正确保持身份的帧数
IDFP = 5      # 身份错误的预测帧数
IDFN = 10     # 丢失的身份帧数
IDF1 = 2×85/(170+5+10) = 0.92

# HOTA (假设)
DetA = 0.85   # 检测精度
AssA = 0.80   # 关联精度
HOTA = √(0.85×0.80) = 0.825
```

---

### 七、指标对比总结

| 维度 | 指标 | 关注点 |
|------|------|--------|
| **综合性能** | MOTA | 检测+关联错误总量 |
| **综合性能** | HOTA | 检测+关联平衡 |
| **身份保持** | IDF1 | ID分配正确性 |
| **定位精度** | MOTP | 框的位置准确度 |
| **轨迹完整性** | MT/ML | 长时跟踪能力 |
| **错误诊断** | FP/FN/IDS/FM | 错误类型分析 |