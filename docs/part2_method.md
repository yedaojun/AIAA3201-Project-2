# Part 2：官方 AnomalyDINO 复现

## 按任务书固定的设置

- 骨干：DINOv2 `dinov2_vits14`，冻结权重，FP32，不微调、不训练分类头。
- 输入：官方短边 448、bicubic resize，底部/右侧裁剪到 patch size 14 的整数倍。
- 数据：VisA 官方 1cls 划分，仅六个必做类别。
- Few-shot：1/2/4/8 正常参考图，seed 0/1/2。
- 参考名单：严格复用 `configs/reference_manifest.json`，与 Part 1 相同。
- 校准：每类最先 20 张正常训练文件独立保留，不能进入参考库；每组阈值是它们的异常分数第 95 百分位。
- agnostic：参考图按官方函数生成 0/45/90/.../315 度旋转，测试图用官方 PCA foreground mask。
- 保留官方默认 `mask_ref_images=False`，即参考库不应用前景屏蔽。不能把额外屏蔽参考背景算作原版行为。
- 最近邻：官方 FAISS CPU IndexFlatL2，归一化特征后 squared L2 / 2，等价于余弦距离。
- 图像分数：官方 `mean_top1p`，取全部 patch 分数的最高 floor(1%) 平均；屏蔽区域分数为零。
- 定位图：官方 `dists2map`，先双线性恢复原图尺寸，再在原图尺度用 scipy Gaussian sigma=4。
- AnomalyCLIP 可选扩展已另行完成，代码在 `Part2/anomalyclip/`，结果在 `results/part2_anomalyclip/`；不混入本节官方 AnomalyDINO 复现结果。

## 哪些是官方代码，哪些是课程适配

`third_party/AnomalyDINO` 和 `third_party/dinov2` 保留官方源码，不修改算法文件。两个 Git commit 与权重 SHA256 都记录在 `results/part2/run_config.json`。

`anomaly_dino/run.py` 直接调用官方 `src.detection.run_anomaly_detection`，并设置任务书要求的参数。
课程适配负责以下工作：

1. 构建独立数据视图 `data/VisA_part2_official/1cls`，利用同磁盘硬链接（不支持时复制）复用原始图片。代码仅只读访问这些图像，不编辑链接文件。
2. 训练视图只包含覆盖本课程全部参考配置的 24 张候选图，使官方原有排序切片正好得到 manifest 的每组参考图。其余正常训练图未用于拟合。20 张校准图只能作为独立查询，使用 `__calibration__` 文件名前缀，不混入参考库。
3. 校准查询和测试查询都经原版算法计算，汇总指标时排除校准查询，仅统计官方测试集。
4. 按 tensor 内容缓存参考旋转特征；查询特征每次新提取并存档，因此正式查询计时包含 GPU 提取。
5. 导出课程要求的逐图 CSV、校准阈值、错误类别、GPU 峰值和汇总结果。

## 原始分数图与精确 Pixel AUROC

保存的是官方 float64 patch 距离 `.npy` 文件，而不是 8 位彩色热力图。调用官方 `dists2map` 可以无损重建正式评估使用的原图分辨率浮点图。
这样避免保存数十 GB 的重复全分辨率 TIFF，同时保留重建所需的全部原始数值。路径直接记录在每条预测的 `anomaly_map` 中。

所有正常测试图的掩码为全零，异常测试图用原始二值缺陷掩码。按类别汇总所有正负像素，保持 float64 精度，分别排序后计算每个正像素胜过负像素的数量，ties 计 0.5。
没有逐图归一化，也没有为了计算 AUROC 量化分数。该实现已与 sklearn 的 AUROC 在随机浮点数据、ties 和相邻浮点值上对照测试。
Part 1 的 256 级实际像素输出使用精确直方图路径；两者使用相同的 AUROC 定义、原图空间、正常图纳入规则和类别平均规则。

## 运行环境

使用项目中的 `.venv-part2`，继承本机 `aiaa3201` 环境已安装的 CUDA torch / torchvision；NumPy 1.26 和官方依赖在独立环境内覆盖，不更改 Part 1 原环境。

```powershell
.\.venv-part2\Scripts\python.exe -m anomaly_dino --categories cashew --shots 1 --seeds 0 --output results/part2_smoke
.\.venv-part2\Scripts\python.exe -u -m anomaly_dino
```

当前依赖清单：`requirements-part2.txt`；实测环境快照：`requirements-part2-tested.txt`。
初次安装该环境时，应先使用正确的 Python 3.11 CUDA 环境创建 venv，再安装对应依赖。骨干从固定的本地官方仓库加载，预训练权重位于 `models/torch/hub/checkpoints/`。

从一份不含数据和第三方目录的新检出复现时，在项目目录按顺序执行（已有环境可跳过安装步骤）：

```powershell
conda create -n aiaa3201 python=3.11 -y
conda activate aiaa3201
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m venv --system-site-packages .venv-part2
.\.venv-part2\Scripts\python.exe -m pip install -r requirements-part2.txt
git clone https://github.com/dammsi/AnomalyDINO.git third_party/AnomalyDINO
git -C third_party/AnomalyDINO checkout b9d1c2648e3a5247437d4d953d907a8f3d994457
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
.\.venv-part2\Scripts\python.exe -c "import torch; torch.hub.set_dir('models/torch/hub'); torch.hub.load('third_party/dinov2','dinov2_vits14',source='local')"
```

再按 Part 1 数据说明准备 `data/VisA_pytorch/1cls` 和参考 manifest，最后运行 `python run_part2.py`。
上述 GPU 包自带 CUDA runtime；需要兼容的 NVIDIA 驱动。本次实测驱动为 576.52，GPU 为 RTX 4060 Laptop 8GB。

## 计时边界

逐图 `inference_seconds` 保留官方计时：图像读取/解码、预处理、骨干特征提取、PCA mask 和 CPU 最近邻搜索，并进行 CUDA synchronize。
官方计时结束位置早于图像分数聚合及全分辨率异常图恢复，输出文件 I/O、像素 AUROC 计算和参考库构建也不计入。
Part 1 原始计时还包括分数聚合与异常图恢复，因此报告中必须明确边界；严格运行速度对比另提供同一批样本、同一完整推理边界的 benchmark。
参考旋转特征缓存仅影响建库时间。GPU peak 指 PyTorch `max_memory_allocated`（包含常驻权重），不是 nvidia-smi 显示的整个驱动进程显存。

## 可复现性和续跑

每组在官方预测完成后保存 `official_scores.json`，随后计算完整像素指标。中断后可复用已经完成的官方预测，或者跳过已完整保存的 `metrics.json` 配置。
实现/配置/manifest/两个官方 commit/权重校验值组成 run signature。修改后应使用新的结果目录，拒绝静默复用不同的旧结果。
全量目标仍为 72 个配置、11,520 次正式测试预测，另外有每组 20 次正常校准查询。

## 原理与需要分析的问题

正常参考图的局部特征形成 memory bank；测试 patch 到最近正常 patch 的距离越大，异常分数越高。
它允许相似正常结构在不同空间位置匹配，通常能减少逐像素模板对位要求带来的干扰。实际收益以同一参考图和测试集上的测量为准。
patch 分辨率有限，小于一个 patch 的细小缺陷可能被周围正常结构稀释；PCA foreground mask 也可能丢弃有用区域。
最终报告应同时查看修复的 FN/FP、剩余错误及新增退步案例，不能只选表现变好的图片。

## 引用

- AnomalyDINO 官方实现：https://github.com/dammsi/AnomalyDINO （WACV 2025）
- DINOv2 官方实现及权重：https://github.com/facebookresearch/dinov2
- VisA / SPot-the-Difference：https://github.com/amazon-science/spot-diff

代码仓库、权重和特征缓存均通过 `.gitignore` 排除；公开提交应按上游许可保留引用和安装说明，而非上传大型数据与缓存。
