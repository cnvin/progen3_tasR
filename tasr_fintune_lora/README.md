# TasR LoRA Fine-tuning + Generation

本目录提供两个脚本：

- `tasr_fintune_lora/train_lora.py`：使用 TasR 蛋白序列对 ProGen3 进行 LoRA 微调。
- `tasr_fintune_lora/generate_with_lora.py`：加载基座模型 + LoRA 适配器进行新序列采样。

依赖：
- 安装本仓库（`bash setup.sh`）会安装 megablocks 与 flash-attn。
- 另外需要 `peft`：`pip install peft`（已在 `pyproject.toml` 中声明）。

数据：
- `tasRdata/protein_sequences.txt`，每行一个蛋白序列（仅 20 个标准氨基酸字母）。

## 微调（LoRA）

示例命令：

```
python tasr_fintune_lora/train_lora.py \
  --base-model Profluent-Bio/progen3-339m \
  --data-file tasRdata/protein_sequences.txt \
  --output-dir outputs/tasr_lora_339m \
  --epochs 2 \
  --batch-size 8 \
  --lr 5e-5 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
```

关键参数：
- `--base-model`：预训练 ProGen3 模型（HF 名称或本地路径）。
- `--data-file`：训练序列文件路径（默认 `tasRdata/protein_sequences.txt`）。
- `--output-dir`：LoRA 适配器保存目录。
- `--precision`：`bf16|fp16|fp32`，建议 GPU 上用 `bf16`。
- `--target-modules`：默认对注意力(`q_proj,k_proj,v_proj,o_proj`)和 MLP(`w1,w2,w3`) 应用 LoRA。

训练脚本会：
- 读取并清洗序列（仅保留 `ACDEFGHIKLMNPQRSTVWY`）。
- 使用 `ProGen3BatchPreparer` 打包 `input_ids / labels / position_ids / sequence_ids`。
- 将 LoRA 注入到指定线性层，进行因果语言建模训练。
- 周期性或每个 epoch 保存 LoRA 适配器权重到 `--output-dir`。

## 生成新序列

示例命令（无条件正向生成）：

```
python tasr_fintune_lora/generate_with_lora.py \
  --base-model Profluent-Bio/progen3-339m \
  --adapter-dir outputs/tasr_lora_339m \
  --output-fasta outputs/tasr_generations.fasta \
  --num-sequences 1000 \
  --min-new 100 --max-new 300 \
  --temperature 0.85 --top-p 0.95
```

要点：
- `--prompt`：默认为 `1`（无条件正向，等价于从 N->C 方向开始）。若要反向无条件，则设为 `2`。
- 生成内部使用仓库自带的 `ProGen3Generator`，自动处理位置编码与终止符。
- 输出为 FASTA 格式，仅保留能还原为有效蛋白序列的采样。

## 备注

- 需要 GPU（建议 A100/H100），并安装 flash-attn（`setup.sh` 已包含）。
- 初次加载大模型可能较慢；如带宽受限，建议先手动拉取模型权重到本地。
- 如果要合并 LoRA 到基座权重，可在推理前使用 PEFT 的 `merge_and_unload()`（不在本脚本中默认启用）。

