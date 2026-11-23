# Project 2 report

`main.tex` is the English report source using the official `cvpr.sty`; `main.pdf` is the updated nine-page report. The report body ends on page 8, within the assignment's six-to-eight-page limit; references continue on page 9. The author block lists Ye Daojun (50027656) and Wang Yiming (50027668), without an affiliation as requested. The full public GitHub URL is printed at the end of the abstract.

The local source set is:

- `main.tex` — report text and tables.
- `references.bib` — cited papers and datasets.
- `cvpr.sty`, `ieeenat_fullname.bst` — files from the [official CVPR author kit](https://github.com/cvpr-org/author-kit).
- `figures/` — diagrams, curves, and qualitative examples used in the report.
- `make_figures.py` — regenerates the figures from the local experiment records.
- `REVIEW_RESPONSE.md` — item-by-item record of the supplied review feedback and changes.
- `REVIEW_COMPLETION_2026-11-23.md` — historical record for the earlier Part 3 review; its old timing figures are superseded.
- `STRICT_REVIEW_V3_RESPONSE_2026-11-24.md` — item-by-item response to the latest strict review and current timing figures.
- `FINAL_REVIEW_V4_RESPONSE_2026-11-24.md` — final five-point correction log and submission audit.
- `evidence/` — compact CSV/JSON audit records suitable for packaging with the report; refresh with `python -m report.export_review_evidence` after local experiments finish.

To rebuild, run `tectonic main.tex` in this directory. A conventional TeX installation can instead run `latexmk -pdf main.tex` (with BibTeX support). The temporary build files, rendered page previews, and local compiler binary are ignored by Git.

The numerical claims are grounded in local Part 1/2/3 experiment records. The primary Part 3 comparison is the pre-fixed A sweep over 1/2/4/8 references and three reference groups. The post-hoc 448+672 scale candidate now has the same 72-setting reference-group coverage but still uses the same VisA test split that motivated it; this is not an independent model-selection result. B/C/E remain single-setting screens. The GitHub link in the conclusion is the repository's configured remote. Ensure the final repository includes the cited code and records before submission.

The earlier diagnostics use `python -m part3.stage_benchmark`, `python -m part3.pixel_ap`, and `python -m part3.diagnostic_ablation`. The follow-up scripts and results are listed in `part3/README.md` and `part3/EXPERIMENT_SEARCH_2026-11-22.md`. The report's three-group A+AnyUp AP values come from `results/part3/a_anyup_pixel_ap_4shot_3seeds.json`; its full shot-count image values come from `results/part3/multiscale_fusion/all_shots_summary.json`. `python -m report.make_followup_tables` writes independent check tables from those records, and `python -m report.verify_followup_numbers` checks the rounded rows inserted in `main.tex`. All post-hoc conclusions retain explicit scope labels.
