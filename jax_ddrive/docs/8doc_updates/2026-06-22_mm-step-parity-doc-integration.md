# 2026-06-22 · 把 mm-SASD-步三方 parity harness 接进文档体系

## 背景
新增了一套**独立**的多模态 SASD 训练步三方 parity harness（`jax_ddrive/scripts/mm_step_parity/`，
新文件、未改任何现有代码），在**完全相同输入**下验证 PyTorch oracle ↔ ddrive_jax NNX ↔ maxtext-fork
在多模态 SASD 训练步上数值一致（含 **MaxText 真·3B 完整前向**）+ 数据 prep→parquet→AR round-trip，
GPU/CPU（一键 `run_overnight_parity.sh` → `SASD_MM_STEP_PARITY_PASS`，10/10）+ **真 v6e-1 TPU**
（`tpu_mm_validate.sh`：data + NNX impl 两样本 PASS）全部通过。

但它最初只记在 harness 自带 README + agent memory 里，**主文档体系 0 处提及**。本次把它接进 living 文档。

## SSOT
权威细节（脚本清单、各子检查、容差、两样本数字、TPU 结果、**ViT cuDNN-Conv3d seed 的根因**
——用对照实验 `Layer1-CTRL`（oracle 图像 embed 走 NNX 路径 → relmax 回到 Layer2 水平）证明那是良性
ViT 种子而非任何一方 bug——以及 jaxtyping staging gotcha）：
**`jax_ddrive/scripts/mm_step_parity/README.md`**。其余文档**只链接、不复制数字**（one-fact-one-home）。

## 改了哪几处（living 文档）
| 文档 | 加了什么 |
|---|---|
| `0overview/02_gotchas.md` §「10 个 gate」 | 在「注」里加一条：mm-步三方 parity 是**独立 harness、未并入 `run_all_verification.sh`**（自带 runner + sentinel）；grep-able marker 列表追加 `SASD_MM_STEP_PARITY_PASS` 及子 sentinel |
| `0overview/01_codebase_map.md` | 「黄金阅读顺序」后的任务分支加一条：验证 mm-SASD-步三方一致 → `scripts/mm_step_parity/`，SSOT 指向 README |
| `to-host-chn.md` | 顶部「最后更新」改 2026-06-22；新增一条 📌 状态横幅（三方 parity + 真 TPU 复核，链 README） |

## 约定说明
- 本条目记录的是**文档体系的变更**（符合本目录用途）；harness 本身是项目产物，其权威记录是它的 README，
  不在 `4collect/` 另起重复文件（避免 one-fact-two-homes）。
- harness 全程**未改动任何现有被跟踪文件**（`git status` 仅显示 `scripts/mm_step_parity/` 为 untracked）。
