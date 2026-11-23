# Project 2 第三版严格审阅逐项处理（2026-11-24）

对照文件：`AIAA3201Project2StrictReviewv3.pdf`。当前成品为 `main.tex` / `main.pdf`：CVPR 样式，正文和结论止于第 8 页，参考文献单列第 9 页。下面的数值都从保存的 CSV/JSON 或本次实测读取。百分比差值用百分点（pp）。

## 1. 六项硬性规范与内部一致性

| 审阅意见 | 处理与核验 |
| --- | --- |
| Section 4 仍写 four shots / seed 0 的 Pixel AP，与 Table 4 三组冲突 | 改为四张参考图、组 0/1/2 的原图尺寸 Pixel AP，明确正常测试图以零 mask 参加。 |
| Table 6 的 178.12 ms 与后文 235/833 ms 属不同计时条件，且缺 672 与峰值显存 | 新增 `part3/unified_benchmark.py`，同一进程、同一 120 张图、预热和 GPU 同步，统一五条查询路径。Table 5（重排后编号）更新为：Official 448 **274.92 ms / 208.28 MiB**，A 448 **361.59 / 210.53**，A+AnyUp **528.93 / 1074.06**，Official 672 **679.71 / 244.33**，448+672 **950.95 / 244.33**。峰值是 PyTorch 已分配显存，含常驻权重，不是设备总占用。融合/448 耗时为 **3.46 倍**。原单独阶段测速仅用于定位 CPU FAISS 开销，不与统一表混作同一次测量。 |
| 多尺度 1/2/4/8-shot 表缺三组样本标准差 | 四档的 Official、A、672、448+672 全改为 mean ± sample SD；由 `all_shots_summary.json` 复核。 |
| Table 3 “blue bold” 不对应实际浅蓝、不利黑白打印 | 所有比较表的强调改为**黑色加粗并加下划线**，表注按实际标识修改。 |
| 官方 CVPR 模板和作者栏 | 项目中的 `cvpr.sty` 是 CVPR author kit 样式，源文件明确规定配合 `article` 类使用，且自动加载 Times；仅凭 `article` 不能判为非官方模板。作者栏仍是组别占位，因组员姓名、学号与正式英文单位尚未提供，按用户先前指示不编造。**正式提交前仍须填写两位作者真实信息。** |
| Figure 3 数值标签与正文、Top-1% 公式 | Figure 3 的 Official DINO 分数改为四位小数 **0.1175**，与正文一致；真实保存分数为 0.1174578905，阈值为 0.1531046269。公式 `k=max(1,floor(0.01P))` 与实现一致。四位小数比 0.118 更精确。 |

## 2. 五项实验逻辑与数据缺口

1. **主线定位。** 报告仍将预先固定的双层特征 A 作为主要受控实验。448+672 是在查看同一测试集错误后设计的后验方向，虽补齐四档、三组，也没有独立验证集，不能把它写成独立验证的最终获胜模型。Section 4 集中说明该界限。
2. **448+672 四向转移。** 四张参考图、三组共 2,880 次判定：`FN→TP=49`、`TP→FN=12`、`FP→TN=19`、`TN→FP=53`。因此异常漏检净少 37、正常误报净多 34，总错判只从 365 降到 362。按六类别的四向表已放入报告；`chewinggum` 与 `pipe_fryum` 的 image AUROC 有退步。逐张记录在 `evidence/all_shots_per_image_transitions.csv`。
3. **200 次 bootstrap 具体结果。** 每个类别/参考组用互斥的 20 张正常校准图。四张参考图下，448 的经验分位/Gaussian/Gumbel 三种阈值平均 bootstrap SD 为 **0.0154/0.0099/0.0094**；测试 FP/FN 为 **154/211、152/194、142/190**。448+672 对应 SD 为 **0.0796/0.0523/0.0512**，测试 FP/FN 为 **188/174、186/178、183/180**。两方法分数单位不同，SD 不横向比较；测试标签仅用于审计结果，不能据此选择分布假设或部署阈值。详细 18 个设置见 `evidence/normal_threshold_estimator_category_seed.csv`。
4. **完整缺陷面积分层。** 报告新增 `<0.5%`、`0.5–2%`、`>2%` 三档，逐档比较 Official 448、A、448+672 的 image AUROC、正常阈值 recall、pixel AP。600 张异常图分为 **439/114/47** 张；大缺陷只涉及四个类别。三档的 Pixel AP 阳性比例不同，只在同一档内做方法比较。
5. **Top-1% 稀释实测。** 新增 `part3/calibrated_aggregation_audit.py`，读取保存的官方 patch 图，分别用 Top-0.1/0.5/1/2% 计算分数，并为每一档重新从 20 张正常校准图取第 95 百分位。四张参考图、三组下 Top-0.5% 的 image AUROC 从 **96.15±0.49%** 到 **96.46±0.64%**，但 FP+FN 从 **365 增到 368**；Top-0.1% 为 **376** 错。`candle bad/000` 和 `capsules bad/087` 在组 0 的四档里都仍是漏检。因此“Top-1% 稀释导致这两例失败”尚未被证实，只能列为可能机理。该筛选使用了同一测试集，不是经过独立验证的改进。

## 3. 四项写作与排版

| 审阅意见 | 处理 |
| --- | --- |
| 公平性声明重复 | Section 4 新设 `Scope and fairness guardrails` 集中交代主线、后验选择、外部监督和人工复核含义；其余位置保留解释具体表格所必需的短句。 |
| Section 6 孤立 6.1 | 改为 6.1 Image scale and defect strata、6.2 Calibration and selective review、6.3 Exploratory directions and limitations。 |
| 表格浮动跨页 | 压缩后验诊断和阈值说明，表格与相关文字紧邻；最后逐页查看第 6–8 页。正文在第 8 页结束，满足 6–8 页正文要求。 |
| Figure 2 inset 和摘要 URL | 调整左图 inset 与图例，重新生成 PNG/PDF 并目视检查；摘要不再放长 URL，结论提供简短的 `Code and results` 可点击链接。 |

## 4. 文件、核验与保留事项

- 报告：`main.pdf`、`main.tex`。脚本：`part3/unified_benchmark.py`、`part3/calibrated_aggregation_audit.py`。
- 审计数据：`evidence/` 下 17 个紧凑 CSV/JSON；原始逐张图和完整异常图仍在本地 `results/`，不放入报告包。
- 已运行：统一 120 张图测速；三组正常校准的 Top-k 重算；`python -m report.verify_followup_numbers` 核对四档 mean±SD 和六类 Pixel AP；Tectonic 编译；渲染目视检查正文末页及关键图表。
- **待用户提供才能完成：** 两位组员英文姓名、学号及正式英文单位。按之前“稍后提供，先保留占位信息”的指示保留占位；此项不能自动填好。
- **学术边界：** 后验多尺度、Top-k 与阈值分布替代都在相同 VisA 测试集上观察过结果。当前报告将其作为探索性审计，后续若要宣称稳定提升，需要独立验证/测试流程。
