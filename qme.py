import os
import json
import torch
import copy
import argparse
from imagenet_utils import test_model_on_dataset, get_model_from_sd
from imagenet_datasets import ImageNet2p
import clip
import torch.nn as nn

class DictParamModule(nn.Module):
    """
    Wraps a dictionary of Tensors into a PyTorch module
    so we can optimize them with PyTorch's optimizer/scheduler.

    - param_dict should be {param_name: Tensor}, with each Tensor requiring grad.
    - We'll store them in a ParameterList, remembering the order of keys.
    """
    def __init__(self, param_dict):
        super().__init__()
        self.param_keys = sorted(param_dict.keys())
        # Create a list of nn.Parameter from the input dictionary
        params = []
        for k in self.param_keys:
            # Make sure it's a float tensor
            p = param_dict[k].float()
            # Wrap as a Parameter so it can be optimized
            p_param = nn.Parameter(p)
            params.append(p_param)
        # Use a ParameterList to store them
        self.params = nn.ParameterList(params)

    def forward(self, checkpoint_dict):
        """
        Compute the "difference loss" between self.params and checkpoint_dict:
        \(\frac{1}{2}\) * sum of squared differences for each parameter.
        """
        loss = torch.tensor(0.0, device=self.params[0].device, requires_grad=True)
        for i, key in enumerate(self.param_keys):
            # Ensure the checkpoint param is on the same device
            ckpt_val = checkpoint_dict[key].to(self.params[i].device).float()
            diff = self.params[i] - ckpt_val
            loss = loss + 0.5 * torch.sum(diff**2)
        return loss

    def to_dict(self):
        """
        Convert the learned parameters back into a standard Python dict
        with the same keys as param_keys.
        """
        out_dict = {}
        for i, key in enumerate(self.param_keys):
            out_dict[key] = self.params[i].detach().clone()
        return out_dict


#########################################################
# 2) Loading raw dicts from 'model_*.pt', ignoring filenames
#########################################################
def load_checkpoints_as_dicts(folder_path, num_checkpoints, device='cpu'):
    """
    Loads all 'model_*.pt' files in lexicographical order,
    extracting a raw param dict from each file (no model wrapping).
    """
    ckpt_files = sorted([
        f for f in os.listdir(folder_path)
        if f.startswith("model_") and f.endswith(".pt")
    ])[:num_checkpoints]

    ckpt_dicts = []
    for filename in ckpt_files:
        full_path = os.path.join(folder_path, filename)
        print(f"Loading {filename} ...")
        checkpoint = torch.load(full_path, map_location=device)

        # If there's a "model_state_dict" key, unwrap it
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        ckpt_dicts.append(checkpoint)

        if device == 'cuda':
            torch.cuda.empty_cache()

    return ckpt_dicts


#########################################################
# 3) Perform "harmonic soup" with a standard PyTorch loop
#########################################################
def uniform_soup(checkpoint_dicts):
    """
    Given a list of raw param dicts, we:
      - Initialize a DictParamModule from the first checkpoint
        (that acts like 'pivot' w_0).
      - For each checkpoint x_i in the list:
          1) difference_loss = sum( (w - x_i)^2 )
          2) backprop
          3) step with LR decaying by 1/i (harmonic)
      => final w after N steps is the uniform average of the N dicts,
         if momentum=0, weight_decay=0, etc.

    Returns a standard Python dict of final parameters.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build our initial "dummy module" from the first checkpoint
    pivot_sd = checkpoint_dicts[0]
    model = DictParamModule(pivot_sd).to(device)
    
    # 2) Create an SGD optimizer with base LR=1
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.0, weight_decay=0.0)

    # 3) Create a scheduler so that after iteration i, LR => 1/(i+1)
    def lr_lambda(step_idx):
        # step_idx is zero-based, so we do 1/(step_idx+1)
        return 1.0 / float(step_idx + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # 4) Iterate over all N checkpoints exactly once
    N = len(checkpoint_dicts)

    model.train()  # set to training mode if you'd like

    # We'll do exactly (N - 1) updates, skipping the pivot checkpoint itself
    num_checkpoints = len(checkpoint_dicts)
    step_idx = 0  # This is used by our scheduler

    # 4) Loop from i=1..(N-1), each iteration i => checkpoint i+1 in 1-based sense
    #    Because pivot is x_1, no need to re-process it.
    for i in range(1, num_checkpoints):
        # Call scheduler.step() BEFORE the optimizer step so iteration i uses lr=1/i
        scheduler.step()  # sets lr=1/(step_idx+1)
        step_idx += 1

        # difference_loss = 1/2‖w - xᵢ‖²
        optimizer.zero_grad()
        loss = model(checkpoint_dicts[i])
        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Iteration {step_idx}, LR={current_lr:.5f}, loss={loss.item():.5f}")

    # Extract final dictionary
    final_sd = model.to_dict()
    return final_sd



def greedy_soup(checkpoint_dicts):
    """
    A 'greedy soup' variant with gradient-based updates:
      - We do multiple epochs over 'checkpoint_dicts'.
      - Each iteration uses difference-loss vs. one checkpoint,
        does a single SG update with LR=1/(step_idx+1).
      - Then we evaluate on `dataset`.
      - If accuracy improves vs. the best known, we keep the new params;
        otherwise revert to the old params.

    Args:
        checkpoint_dicts (list[dict]):
          Each is a state_dict for the same architecture.
        dataset:
          An ImageNet2p dataset or similar for validation.
        test_model_fn(model, dataset) -> float:
          Returns validation accuracy (or any metric).
        num_epochs (int):
          How many passes over all checkpoints.
        pivot_idx (int):
          Which checkpoint is pivot. We'll initialize from checkpoint_dicts[pivot_idx].

    Returns:
        dict: final state_dict after greedy acceptance steps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_cls = ImageNet2p
    data_location = '/scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/model-soups/data'
    batch_size = 256
    workers = 4

    base_model , preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    print('Preparing ImageNet2p')
    dataset = dataset_cls(preprocess, data_location, batch_size, workers)


    # 1) Build DictParamModule from pivot
    pivot_sd = checkpoint_dicts[0]
    model = DictParamModule(pivot_sd).to(device)
    model.train()

    # 2) Create an SGD optimizer with base LR=1.0, no momentum, no WD
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.0, weight_decay=0.0)

    # 3) LambdaLR => LR=1/(step_idx+1)
    def lr_lambda(step_idx):
        return 1.0 / float(step_idx + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    vit_model = get_model_from_sd(model.to_dict(), base_model)

    # Evaluate initial pivot accuracy
    best_acc = test_model_on_dataset(vit_model, dataset)
    print(f"Initial pivot accuracy = {best_acc:.3f}%")

    N = len(checkpoint_dicts)
    global_step = 0

    model.train()  # set to training mode

    for i, ckpt_sd in enumerate(checkpoint_dicts):
        # Save old model parameters using state_dict to preserve key names
        old_params_sd = copy.deepcopy(model.state_dict())

        optimizer.zero_grad()
        loss = model(ckpt_sd)  # forward pass: computes difference loss
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        global_step += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Iteration {global_step}, loss={loss.item():.6f}, LR={current_lr:.6f}")

        # Evaluate new model on dataset
        vit_model = get_model_from_sd(model.to_dict(), base_model)
        new_acc = test_model_on_dataset(vit_model, dataset)
        if new_acc >= best_acc:
            print(f"    Improved accuracy: {new_acc:.3f}% (old={best_acc:.3f}%). Keeping update.")
            best_acc = new_acc
        else:
            print(f"    Accuracy not improved: {new_acc:.3f}% (best={best_acc:.3f}%). Reverting.")
            with torch.no_grad():
                for name, param in model.state_dict().items():
                    param.copy_(old_params_sd[name])  # revert in place

    final_sd = model.to_dict()
    return final_sd



def qme(
    checkpoint_dicts,
    num_epochs=5,
    optimizer_name="sgd",
    lr=1e-3,
    weight_decay=0.0,
    eps=1e-8,
    beta1=0.9,
    beta2=0.999
):
    """
    Multi-epoch training on the difference-loss:
       sum_over_params( (w - x_i)**2 ),
    across 'checkpoint_dicts'.

    Args:
        checkpoint_dicts (list[dict]):
            List of raw param dicts. The first is the "pivot."
        num_epochs (int):
            How many epochs to run over the entire list.
        optimizer_name (str):
            One of ['sgd', 'adamw', 'adagrad'].
        lr (float): Learning rate (η).
        weight_decay (float): Weight decay (λ).
        eps (float): Smoothing term (ε).
        beta1 (float): Momentum / beta1.
        beta2 (float): Beta2 for AdamW.

    Returns:
        final_sd (dict): final dictionary of parameters after multi-epoch training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Create a DictParamModule from the first checkpoint
    pivot_sd = checkpoint_dicts[0]
    
    #or pivot_sd can be the uniform soup checkpoint
    # pivot_sd = torch.load('/scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ansharora/Quadratic-Model-Ensembling/imagenet_ensembled_models/qme_greedy_soup_final.pt', map_location=torch.device('cpu'))['model_state_dict']
    
    model = DictParamModule(pivot_sd).to(device)
    model.train()

    # 2) Build the chosen optimizer
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=beta1,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            betas=(beta1, beta2)
        )
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    else:
        raise ValueError("optimizer_name must be in ['sgd','adamw','adagrad']")

    # 3) Set up harmonic learning rate scheduler
    def lr_lambda(step_idx):
        return 1.0 / (step_idx + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0
    N = len(checkpoint_dicts)

    # 4) Multi-epoch loop
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        for i, ckpt_sd in enumerate(checkpoint_dicts):
            optimizer.zero_grad()
            loss = model(ckpt_sd)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Iter={global_step}, Loss={loss.item():.5f}, LR={current_lr:.6f}")

    # 5) Return final dictionary
    final_sd = model.to_dict()
    return final_sd




#########################################################
# 4) Main script
#########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=False,
                        help="Folder containing 'model_*.pt' files.")
    parser.add_argument("--ranks_file", type=str, required=True,
                        help="JSON/JSONL with {\"ranks\":[...]} for reordering.")
    parser.add_argument("--out_path", type=str, default="harmonic_soup_sgd.pt",
                        help="Where to save final soup checkpoint.")
    parser.add_argument("--type", type=str, required=False,
                        choices=['uniform_soup', 'greedy_soup', 'qme'], default = 'uniform_soup')
    parser.add_argument("--optimizer", type=str, required=False,
                        choices=['sgd', 'adamw', 'adagrad'], default = 'sgd')
    parser.add_argument("--num_epochs", type=int, default = 40)
    parser.add_argument("--num_checkpoints", type=int, default = 72)
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--eps", type=float, default=1e-9,
                        help="Smoothing term (AdamW/Adagrad).")
    parser.add_argument("--beta1", type=float, required=False,
                        help="Momentum param")
    parser.add_argument("--beta2", type=float, required=False,
                        help="2nd momentum param (beta2 in AdamW).")
    
    args = parser.parse_args()

    # args.ranks_file = '/scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ansharora/Quadratic-Model-Ensembling/imagenet_ckpts_ranks_imagenetv2.jsonl'
    # args.folder = '/scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/model-soups/models'
    # args.out_path = f'/scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ansharora/Quadratic-Model-Ensembling/imagenet_ensembled_models/{args.out_path}'
    # args.out_path = '/scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ansharora/Quadratic-Model-Ensembling/imagenet_ensembled_models/qme_adamw_1_40ep.pt'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load raw checkpoint dicts
    ckpt_dicts = load_checkpoints_as_dicts(args.folder, args.num_checkpoints, device=device)
    N = len(ckpt_dicts)
    if N == 0:
        print("No checkpoints found. Exiting.")
        return
    print(f"Loaded {N} checkpoint dicts from {args.folder}.")

    # 2) Optionally reorder by ascending rank
    if args.ranks_file and os.path.exists(args.ranks_file):
        with open(args.ranks_file, 'r') as f:
            data = json.loads(f.readline())
        ranks = data["ranks"]
        if len(ranks) != N:
            raise ValueError(f"ranks length {len(ranks)} != number of checkpoints {N}.")


        ##Ascending order##
        # Pair each dict with its rank, then sort #Order1 -> Default
        # zipped = list(zip(ckpt_dicts, ranks))
        # sorted_zip = sorted(zipped, key=lambda x: x[1])
        # ckpt_dicts = [x[0] for x in sorted_zip]
        # sorted_ranks = [x[1] for x in sorted_zip]
        # print("Reordered by ascending rank. First few ranks:", sorted_ranks[:5])
        
        ##Descending order##
        zipped = list(zip(ckpt_dicts, ranks))
        sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
        ckpt_dicts = [x[0] for x in sorted_zip]
        sorted_ranks = [x[1] for x in sorted_zip]
        print("Reordered by descending rank. First few ranks:", sorted_ranks[:5])

    if args.type == 'uniform_soup':
        # 3) Perform harmonic soup via PyTorch's SGD+LambdaLR
        print("Performing harmonic soup with standard training loop (SGD+scheduler)...")
        final_sd = uniform_soup(ckpt_dicts)
    elif args.type == 'greedy_soup':
        final_sd = greedy_soup(ckpt_dicts)
    elif args.type == 'qme':
        final_sd = qme(ckpt_dicts, num_epochs=args.num_epochs, optimizer_name=args.optimizer, lr=args.lr,
                       weight_decay=args.weight_decay, eps=args.eps, beta1=args.beta1, beta2=args.beta2)

    # 4) Save final
    out_obj = {"model_state_dict": final_sd}
    torch.save(out_obj, args.out_path)
    print(f"Saved final harmonic soup to {args.out_path}")


if __name__ == "__main__":
    main()



# python compare_soup.py \
#   --input_1 /scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ansharora/QME/qme_uniform_soup_bert_sst2.pt \
#   --input_2 /scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ansharora/Quadratic-Model-Ensembling/averaged_state_dict_bert_sst2.pth \
#   --model bert \
#   --soup_type uniform
    


# python qme.py --type uniform_soup --out_path  qme_saved_models/9C_10_qme_uniform.pt --folder /scratch3/workspace/oraundale_umass_edu-quadratic-ensembling/ojas/QME/saved_models/9C_10_Meta_CIFAR10_vit_25Epk --ranks_file None