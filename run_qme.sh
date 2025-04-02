##Uniform Soup
python3 qme.py \
    --type uniform_soup \
    --out_path qme_uniform_soup_try.pt \
    --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl \
    --folder model-soups/models \
    --num_checkpoints 72
    
    


# ##Greedy Soup
# python3 qme.py \
#     --type greedy_soup \
#     --out_path qme_greedy_soup_final.pt \
#     --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl \
#     --folder model-soups/models \
#     --num_checkpoints 72


# ##QME-AdamW
# python3 qme.py \
#     --type qme \
#     --optimizer adamw \
#     --lr 1e-4 \
#     --beta1 0.8 \
#     --beta2 0.9 \
#     --weight_decay 0.01 \
#     --eps 1e-9 \
#     --num_epochs 25 \
#     --out_path qme_adamw_soup_hyp1_try.pt
#     --folder model-soups/models \
#     --ranks_file imagenet_ckpts_ranks_imagenetv2.jsonl
#     --num_checkpoints 72

