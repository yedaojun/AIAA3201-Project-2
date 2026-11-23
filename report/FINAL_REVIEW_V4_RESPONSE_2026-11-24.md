# Project 2 终审 v4 修改与提交核验

审阅文件：`aiaa3201_project2_final_review_v4.md`。本轮遵循“实验封板”建议，没有增加模型实验；只修正文稿与准备必交材料。

| v4 意见 | 本轮处理 |
| --- | --- |
| 摘要末尾缺显式 GitHub URL | 恢复完整 `https://github.com/yedaojun/AIAA3201-Project-2`，置于 abstract 最后一句。已检查 PDF 第 1 页，完整 URL 可见且可点击。 |
| CVPR 模板与作者占位 | `main.tex` 明确使用 `\documentclass[times,10pt,twocolumn,letterpaper]{article}` 和 `\usepackage[final]{cvpr}`。项目内的官方 `cvpr.sty` 明确要求配合 `article` 类，并自动加载 `times`。填入用户提供的 Ye Daojun（50027656）、Wang Yiming（50027668）；按用户要求不写学院。 |
| 效率节遗漏 CPU RGB 模板耗时 | Section 5.3 加入原单独基准的 **28.21 ms/image、0 MiB GPU**，并明确它与五条 GPU 查询路径的统一计时边界不同。 |
| 1%/5% FPR 免责句悬空 | 删除未在正文配具体 FNR 的句子；已有 bootstrap 与人工复核结论保持。 |
| Figure 3 四位小数不统一 | RGB 改为 **0.5202 / 0.4937**；DINO 仍为 **0.1175 / 0.1531**，图文一致。图由 `make_figures.py` 再生成。 |

## 必交材料

- 报告：`main.pdf`，9 页，其中正文及结论 8 页，参考文献第 9 页。
- 仓库：`https://github.com/yedaojun/AIAA3201-Project-2`。GitHub 公共 API 以未登录请求返回 `private=false, visibility=public`；仓库 README 说明依赖、VisA 数据准备、权重来源、运行/评估命令，并展示代表性图片。
- 演示包：项目根目录 `anomaly_demo.zip`，含 `anomaly_demo.mp4`、三张实际单图推理输出、逐例分数 CSV 与可重现命令。它用保存的四张参考图/group 0 双层 DINOv2 输出制作，展示正常 OK、缺陷 NG、剩余漏检三个例子；不是新模型实验或交互界面的实录。ZIP 经 `testzip()` 校验，MP4 为 1280×720、20 fps、25 秒。

## 核验范围

- Tectonic 编译成功；PDF 首末页和 Figure 3 已目视检查，正文仍在 8 页内；日志无未定义引用或 Overfull 警告。
- `python -m report.verify_followup_numbers` 已核对报告中四档 mean±SD 与六类精确 Pixel AP。
- `anomaly_demo.zip` 的视频和例图均从真实单图输出生成；CSV 分数与正常校准阈值一致。演示使用数据路径中的测试类别只作事后说明，推理代码不读取标签或 ground-truth mask。

本轮未将探索性 448+672 或 AnyUp 重新定义为预先锁定的主方法；报告继续把 Part 3 A 作为受控主比较，把其余方向标为后验或不同监督条件的补充。
