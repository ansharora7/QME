# 🥣 Quadratic Model Ensembling (QME)

This repo helps create a model soup — a combination of multiple model checkpoints — using Quadratic Model Ensembling. It supports optimization using `SGD`, `AdamW`, or `Adagrad`.

## 🧪 Environment Setup

It's recommended to use the following conda environment:

```bash
conda activate /project/pi_wenlongzhao_umass_edu/18/ansharora/.conda/envs/model_soups
```

## 🔧 Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--folder` | `str` | Folder containing the `model_*.pt` checkpoint files. |
| `--ranks_file` | `str` | Path to a JSON or JSONL file containing a `{"ranks": [...]}` array for checkpoint ordering. |
| `--out_path` | `str` | Path to save the final soup checkpoint. |
| `--type` | `str` | Strategy for combining models. Options: `uniform_soup`, `greedy_soup`, `qme`. |
| `--optimizer` | `str` | Optimizer to use. Options: `sgd`, `adamw`, `adagrad`. |
| `--num_epochs` | `int` | Number of optimization epochs. |
| `--num_checkpoints` | `int` | Number of model checkpoints to load. |
| `--lr` | `float` | Learning rate for the optimizer. |
| `--weight_decay` | `float` | Weight decay (L2 regularization). |
| `--eps` | `float` | Epsilon term for numerical stability (AdamW/Adagrad). |
| `--beta1` | `float` | Momentum term (e.g., for AdamW). |
| `--beta2` | `float` | Second moment decay rate (used in AdamW). |

## 🏃‍♂️ Example Usage

You can refer to `run_qme.sh` for runs. Below are some common use cases:

### 🍜 Uniform Soup

```bash
python3 qme.py \
    --type uniform_soup \
    --out_path qme_uniform_soup_try.pt \
    --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl \
    --folder model-soups/models \
    --num_checkpoints 72
```

### 🔍 Greedy Soup
```bash
python3 qme.py \
    --type greedy_soup \
    --out_path qme_greedy_soup_final.pt \
    --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl \
    --folder model-soups/models \
    --num_checkpoints 72
```

### ⚙️ QME with AdamW Optimizer
```bash
python3 qme.py \
    --type qme \
    --optimizer adamw \
    --lr 1e-4 \
    --beta1 0.8 \
    --beta2 0.9 \
    --weight_decay 0.01 \
    --eps 1e-9 \
    --num_epochs 25 \
    --out_path qme_adamw_soup_hyp1_try.pt \
    --folder model-soups/models \
    --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl \
    --num_checkpoints 72
```

### ⚙️ QME Batching with AdamW Optimizer
```bash
python3 qme.py \
    --type qme_batch \
    --optimizer adamw \
    --lr 1e-4 \
    --beta1 0.8 \
    --beta2 0.9 \
    --weight_decay 0.01 \
    --eps 1e-9 \
    --num_epochs 25 \
    --batch_size 4 \
    --out_path qme_adamw_soup_hyp1_try1.pt \
    --folder model-soups/models \
    --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl \
    --num_checkpoints 72
```


## ⚠️ Warnings

Please review the following caveats before running the code:

### 📄 `qme.py` — `load_checkpoints_as_dicts`

- **Line 67**: The function looks for checkpoint files starting with `"model_"`.  
  🔧 *If your checkpoints follow a different naming pattern (e.g., `"ckpt_"`, `"checkpoint_"`), you must update this line accordingly:*

  ```python
  if f.startswith("model_") and f.endswith(".pt")
  ```

- **Line 77**: The script assumes the checkpoint dictionary has a top-level key `"model_state_dict"`.
🔧 If your checkpoints are saved with a different key or directly as state dicts, modify the following block:

   ```python
   if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
   ```

### 🔍 `greedy_soup` mode

- The **greedy soup** strategy is currently implemented **only for ImageNet**.  
  This is because the associated evaluation logic and re-ranking methods are hardcoded for ImageNet-style validation.

  ❗ *If you plan to use greedy soup on another dataset, you’ll need to modify the evaluation and ranking mechanisms accordingly.*
