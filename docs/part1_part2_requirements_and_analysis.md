# Part 1 / Part 2：要求核对与关键分析

更新：2026-11-16。依据课程任务书第 3 页 §5.1、§5.2，以及第 4 页 §6.2。

**结论：Part 1 / Part 2 的核心实现、规定实验和两个选做扩展已完成。此前展示材料对分析的表达不足，本次将原因解释补入报告、PPT 正文和英文配音稿。本文是阶段报告补充，不是最终 6–8 页 CVPR 格式 PDF。Part 3 仍未完成。**

## 1. 逐项核对

**规则来源说明：**“前 20 张正常训练图 + 第 95 百分位”明确写在任务书 §5.1 Part 1 的 **Demo Threshold** 中。§5.2 Part 2 明确要求参考采样时排除这些校准图，但没有再次指定 AnomalyDINO 必须采用 95 分位阈值。我们为 Part 2 Demo 的 OK/NG 输出额外沿用同一正常校准策略；这是本项目的决策层设计，不是官方 AnomalyDINO 评分方法或 Part 2 明文规定的阈值。校准不修改官方原始分数或异常图；Image/Pixel AUROC 仍使用阈值前的分数，Accuracy、FP/FN 则是在本项目该决策策略下报告。


| 任务书要求 | 核对结果 | 可追溯证据 |
|---|---|---|
| 六类 VisA、官方 1cls 划分、全部测试图与掩码 | 已完成：360 正常 + 600 异常 | `configs/reference_manifest.json`；`results/part2/audit.json` |
| Part 1 使用与 AnomalyDINO 相同的正常模板、相同输入尺寸，选择最近模板 | 已完成：RGB 模板，短边 448，保持比例、裁至 14 的倍数 | `anomaly_baseline/core.py`；两种方法的 `runs.csv` |
| 像素差异图和后处理 | 已完成 RGB 绝对差和高斯平滑；未加形态学、ECC 配准 | `anomaly_baseline/core.py`。任务书允许技术合理的路线调整；这些取舍须披露，不能写成已完成了形态学或配准 |
| Part 1 明确规定：排序前 20 张正常图、95 分位 Demo 阈值；Part 2 Demo 沿用此策略 | 已完成；参考库、模型拟合均排除校准图 | manifest；每组 `calibration_images_json`、阈值和输入角色记录 |
| Part 1 Key Analysis：姿态、背景、光照为何被误当缺陷 | 本次明确补全，见第 2 节 | 正常 capsules `good/016.JPG` 的误报为实测证据；背景、光照是机理解释，没有声称做了受控扰动实验 |
| Part 1 选做自编码器：正常图训练、重建误差、单列训练数 | 已完成；每个种子 2,813 拟合 + 312 验证，六类各 20 轮；种子 0/1/2 | `results/part1_autoencoder_multiseed/`；不是 few-shot 比较 |
| Part 2 官方代码、`dinov2_vits14`、448、agnostic | 已完成；官方源码和权重来源已记录 | `results/part2/run_config.json`；`anomaly_dino/official.py`；`results/part2/audit.json` |
| 正常 patch memory bank、最近正常邻居评分 | 已完成；冻结 backbone，归一化特征最近邻 | 官方调用和 `anomaly_dino/predict.py` |
| 1/2/4/8 张参考图，各 seed 0/1/2 | 两个必做方法各 72 个类别配置、11,520 次正式测试预测 | 两种方法 `runs.csv`、`predictions_all.csv` |
| 保存、复用参考文件名，采样前排除校准图 | 本轮重新配对检查全部 72 个配置：参考名单一致、校准图不相交 | `results/requirements_learning_audit.json` |
| 未来 improvement 也复用这些参考图 | 名单已固定可复用；实际改善实验尚未开始，暂不能标为已验证 | `configs/reference_manifest.json`；Part 3 后续工作 |
| Part 2 Key Learning Point：冻结语义特征为何能定位未见缺陷 | 本次补全“预训练迁移—正常库—距离—空间定位”的解释 | 第 3 节；新版 PPT 第 4 页；视频 2:22–2:40 |
| Part 2 Key Learning Point：粗 patch 分辨率在哪里失败 | 本次补全原始网格、稀疏缺陷和推理边界说明 | 第 4 节；新版 PPT 第 9 页；视频 4:34–4:54 |
| Part 2 选做 AnomalyCLIP：使用适用于 VisA 的官方 checkpoint，披露外部训练数据 | 已完成：MVTec 训练的 `_visa/epoch_15.pth`；无 VisA 拟合或参考库，另用正常图校准 | `docs/optional_extensions.md`；`results/part2_anomalyclip_multiseed/`。square 448 与必做保持比例的几何不同，单独报告 |
| Image AUROC：逐类及六类等权均值；用阈值前分数 | 已完成 | `results/part1/summary_by_category.csv`、`results/part2/summary_by_category.csv` |
| Pixel AUROC：纳入正常全零掩码，逐类算后等权平均 | 已完成；未逐图归一化评估分数 | `results/part2/audit.json`；原始浮点 patch 图和评估实现 |
| 每种 shot 的三组均值和标准差 | 已完成；sample SD，先组内六类平均再跨组统计 | 两种必做方法 `summary_macro.csv`；`output/video/method_summary.csv` |
| 同设备平均推理耗时、峰值 GPU 内存、说明特征提取 | 已有串行同输入 benchmark，见第 5 节 | `docs/RESULTS.md`；`docs/published_results/` 的 benchmark 表；不混用不同计时边界 |
| 五个有代表性的困难案例及实际错误解释 | 已有五个困难案例，另有固定 22 个配对案例 | `results/part2/RESULTS.md`、`difficult_cases.csv`、`fixed_case_comparison.csv` |

其中 seeds 0/1/2 沿用官方排序切片的参考组定义，并不是另行声明做了随机抽样。输入保持纵横比，因此“448”是短边，不代表每张输入都为 448×448。

## 2. Part 1 Key Analysis：为什么正常变化会触发差分？

### English report text

The template baseline compares RGB values at corresponding image coordinates. A change in pose or product arrangement moves normal boundaries, so pixels containing an object in one image can correspond to background in the other. Background texture changes also produce residuals unrelated to a product defect. Illumination changes alter intensity or color even when the physical surface is unchanged. Gaussian smoothing suppresses small fluctuations but does not establish correspondence or illumination invariance.

Our normal capsules image `good/016.JPG` illustrates the alignment problem. At four references and group 0, the template score is 0.5202, above its normal-only threshold of 0.4937: a false positive. The saved test/template comparison shows different normal layouts and responses across ordinary capsule regions. On the same image, AnomalyDINO scores 0.1175 below its own threshold of 0.1531, correcting this false alarm. This is observational support for reduced alignment sensitivity, not an isolated measurement of the effect of pretraining or augmentation.

Threshold adjustment cannot repair an unsuitable representation. Lowering the threshold may recover a low-scoring defect while also rejecting more normal images. The overlapping score distributions and weak template Image AUROC provide a broader explanation than a single selected example. We did not run controlled pose, background, or illumination perturbation experiments, so we report these mechanisms without claiming measured robustness to them.

### 中文理解

模板比较的是**同一坐标的像素**，而不是“同一种正常结构”。产品移动后，原本是物体的坐标可能变成背景；背景纹理或照明变化也会造成差异。平滑能减少噪声，但不能自动解决对齐或光照问题。

正常胶囊图的误报有实际分数支持。背景、光照的解释来自算法机理，不能写成已分别做过控制变量实验。降低阈值只改变误报与漏报的取舍，不能让正常/异常分数本身更可分。

![实际正常输入、模板与响应](../results/part1/supplement/template_comparisons/capsules_FP.png)

## 3. Part 2 Key Learning Point：为什么冻结特征能发现未见缺陷？

### English report text

DINOv2 learns transferable visual representations during large-scale self-supervised pretraining. In our experiment its weights remain fixed. A normal reference bank therefore supplies product-specific examples of normal appearance without learning a new defect classifier. For a test patch descriptor, AnomalyDINO retrieves its closest normal descriptor and uses their distance as an anomaly score. A defect can receive a high score without ever having appeared as a labeled training example, because the rule asks whether the patch resembles observed normal structure rather than which defect class it belongs to.

Searching across the normal bank permits a normal local structure to match at a different image location. This reduces the strict pixel alignment requirement of RGB template differences. The test patch retains its own spatial location, so placing the distances back on the patch grid gives a localized anomaly map. This explains the possibility of both unseen-defect detection and localization using a frozen backbone. It does not guarantee that every defect has a distinct representation or that every legitimate normal variation is covered by a small bank.

Our paired cases support the practical value of this system. Capsules `bad/000.JPG` changes from FN to TP, while the normal layout example changes from FP to TN. These observations and the full-set AUROC improvement are evidence of the complete method's behavior. They do not separate the effects of pretraining, reference rotation, PCA masking, and nearest-neighbor matching. “Training-free” means no task-specific backbone optimization on VisA; it does not mean that DINOv2 has never been pretrained.

### 中文理解

1. **能力来自已有预训练。** DINOv2 事先学到了可迁移的局部外观与结构表征。
2. **正常图定义本类别的正常范围。** memory bank 告诉系统哪些局部特征在正常产品中出现过。
3. **不用预先见过某种缺陷。** 测试区域找不到足够近的正常邻居，就可能被判为异常。
4. **保留空间位置就能定位。** 每个测试 patch 的距离放回其原有网格位置，形成热力图。
5. **允许跨位置匹配。** 它比逐像素差分少依赖严格对齐，但不意味着对姿态、背景、照明完全不变。

![固定胶囊缺陷的配对比较](../results/part2/examples/fixed_07_capsules_FN_to_TP_zoom.png)

来源：[AnomalyDINO 论文](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)、[DINOv2 官方模型说明](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)。上述本地数值来自已保存的预测，而非论文的 benchmark 数字。

## 4. Where coarse patch resolution fails

### English report text

ViT-S/14 samples the input on a 14-pixel patch grid. After the prescribed resize and crop, our candle image `bad/000.JPG` has a 490×448 input and a saved 32×35 anomaly-score grid. A descriptor combines evidence from its patch and contextual transformer processing; its effective context is not restricted to a hard 14×14 receptive field. Thin or sparse damage can have a weak influence when mixed with normal structure. Restoring this coarse score grid to the original resolution produces a smooth map, but interpolation and Gaussian smoothing cannot create missing feature evidence or guarantee precise defect boundaries.

This image's annotation occupies 0.095018% of the original image. As an explicitly post-hoc geometric diagnostic, nearest-neighbor resizing of its binary mask gives a 20×22 bounding box spanning six patch cells; even the most occupied cell contains only 39.8% labeled defect pixels. Thus, the entire defect is **not** smaller than one patch. The more accurate concern is sparse, mixed-content sampling. The diagnostic mask was not supplied to inference or used to choose settings.

At four references/group 0, the image score is 0.147887, below the independent normal-only threshold of 0.176560. This establishes an image-level false negative. It does not prove that patch resolution alone caused the error: feature similarity, foreground masking, top-1% aggregation and threshold calibration are alternative or interacting explanations. A bright spot in a zoomed heatmap would likewise not prove a correct image-level decision. The observed sparse geometry and coarse map motivate a controlled feature-layer or scale comparison in Part 3; such an ablation has not yet been performed.

### 中文理解

**已经观察到：**小而稀疏的真实缺陷、32×35 原始分数网格、低于阈值的图像分数，以及实际 FN。

**可以解释的机理：**缺陷与正常内容混合，局部特征的差异可能不够强；将低分辨率热力图放大不能补回没有提取到的细节。

**尚不能断言：**这个 FN 只由 patch 分辨率造成。没有做改变尺度或特征层的受控实验，不能提前把假设写成实验证明。也不能声称该缺陷包围框小于 14×14。

![当前未解决的 candle 漏检：输入、GT、模板图、DINO 图](../results/part2/examples/fixed_03_candle_FN_to_FN_zoom.png)

几何诊断和参考核对的完整数值：`results/requirements_learning_audit.json`。

## 5. 效率、五个困难案例与解释边界

同一 RTX 4060 Laptop 8GB，每种方法串行处理固定 120 张输入。时间包括解码、预处理、特征提取/匹配、图像评分和原图热力图恢复，不含建库、训练、保存和指标计算。配置为必做方法 4-shot/group 0；输入短边 448。GPU 指峰值 allocated tensor memory，不是进程总显存。

| 方法 | 平均 ms/image | 峰值 GPU MiB |
|---|---:|---:|
| RGB template | 28.21 | 0 |
| AnomalyDINO | 211.77 | 120.48 |
| Autoencoder | 48.47 | 26.81 |
| AnomalyCLIP | 222.10 | 1,879.25 |

以上是已有同边界 benchmark，不是本次新增训练或测速。早期官方查询计时与端到端计时的边界不同，不宜混用。

以下五个困难案例读取既有预测，并逐张检查输入、掩码、热力图和叠加图。它们不仅是五张图片，还对应不同的错误现象。所有分数均为 4-shot/group 0，阈值未按测试结果修改。

| 案例 | 已观察到的错误与热力图现象 | 分析及其边界 |
|---|---|---|
| candle `good/0400.JPG` | 正常、全黑 GT；0.2547 > 0.1766，为新增 FP。正常烛面上有局部响应 | 正常局部外观也可能远离小参考库。不能把热力图亮点当成真实缺陷；是否由照明或参考覆盖造成需要另测 |
| capsules `bad/087.JPG` | 很稀疏的缺陷标注；0.1345 < 0.1531，由模板 TP 变为 DINO FN；整图响应较弱 | 语义相似不保证保留细微损伤差异。模板 TP 也不证明其热力图正确定位；两种方法各有退步案例 |
| cashew `bad/063.JPG` | GT 是很小区域；0.1243 < 0.1819，仍为 FN；响应较分散，没有强烈突出标注处 | 微小缺陷弱响应与较高校准阈值共同影响决策。局部特征、前景筛选、分数汇总可能交互，不能只归因于尺寸 |
| chewinggum `good/342.JPG` | 正常、全黑 GT；0.2599 > 0.1657，为新增 FP；正常产品内部出现局部亮条 | 预训练特征不是对所有正常纹理/反光都不变。当前显示实际误报，不声称已验证具体照明原因 |
| fryum `bad/031.JPG` | 极小 GT；0.1105 < 0.1275，仍为 FN；响应覆盖较大正常结构而未形成足够强的缺陷区分 | 粗局部表征可能缺乏细微变化的对比度；整图阈值和 top-1% 汇总也要分开研究 |

![正常 candle 的新增 FP](../results/part2/examples/difficult_candle_FP.png)

![capsules 的新增 FN](../results/part2/examples/difficult_capsules_FN.png)

![cashew 的残余 FN](../results/part2/examples/difficult_cashew_FN.png)

![正常 chewinggum 的新增 FP](../results/part2/examples/difficult_chewinggum_FP.png)

![fryum 的残余 FN](../results/part2/examples/difficult_fryum_FN.png)

完整分数与图片路径还见 `results/part2/difficult_cases.csv`。另有固定 22 个从 Part 1 沿用的配对案例，包含改善、保留错误和退步，没有只选择变好的图片。

## 6. 本次补到了哪里？

| 材料 | 内容位置 |
|---|---|
| 本文及流程总结 | Part 1 机理与阈值；Part 2 预训练迁移与空间定位；粗网格的实测诊断与因果边界 |
| 新版 PPT 第 3 页 | 正常布局误报，加上姿态、背景、照明的差分机理 |
| 新版 PPT 第 4 页 | 为什么冻结特征可以检测未见缺陷 |
| 新版 PPT 第 9 页 | patch 网格局限、candle FN、下一阶段对照实验 |
| 视频 0:32–0:54 | Part 1 差分为何混淆正常变化 |
| 视频 2:22–2:40 | 独立分析页：预训练迁移、正常 bank、未见缺陷距离和空间定位。官方配置单列在 2:04–2:22 |
| 视频 4:34–4:54 | 稀疏小缺陷、原始网格与尚待验证的原因 |
| 双语讲稿 | 上述视频三段均有对应英文及中文，不能只在备注里补分析 |

## 7. 仍需区分的最终提交事项

当前结论针对 Part 1 / Part 2 的阶段检查。整个课程项目仍要求 Part 3 实际改善及与未修改 Part 2 的受控比较、最终 6–8 页 CVPR 格式 PDF、完整引用、公开仓库同步，以及 `anomaly_demo.zip` 等最终打包。它们不能因为阶段实验已经完成而自动标为全部完成。中期五分钟视频可以报告当前结果和明确的下一阶段计划，不需要把尚未开展的实验写成已完成。
