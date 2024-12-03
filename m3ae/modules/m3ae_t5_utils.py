import torch
import torch.nn as nn
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import AdamW

from .objectives import compute_irtr_recall 
from ..gadgets.my_metrics import VQARADScore, Accuracy, Scalar, ROUGE1Score, ROUGE2Score, BLEUScore

def set_metrics(pl_module):
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams.m3ae_config["loss_names"].items():
            if v <= 0:
                continue
            if split == "train":
                setattr(pl_module, f"train_{k}_score", VQARADScore())
                setattr(pl_module, f"train_{k}_rouge1", ROUGE1Score())
                setattr(pl_module, f"train_{k}_rouge2", ROUGE2Score())
                setattr(pl_module, f"train_{k}_bleu_score", BLEUScore())
                setattr(pl_module, f"train_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"val_{k}_score", VQARADScore())
                setattr(pl_module, f"val_{k}_rouge1", ROUGE1Score())
                setattr(pl_module, f"val_{k}_rouge2", ROUGE2Score())
                setattr(pl_module, f"val_{k}_bleu_score", BLEUScore())
                setattr(pl_module, f"val_{k}_loss", Scalar())

                setattr(pl_module, f"test_{k}_score", VQARADScore())
                setattr(pl_module, f"test_{k}_rouge1",ROUGE1Score())
                setattr(pl_module, f"test_{k}_rouge2",ROUGE2Score())
                setattr(pl_module, f"test_{k}_bleu_score", BLEUScore())
                setattr(pl_module, f"test_{k}_loss", Scalar())

def epoch_wrapup(pl_module, test=False):
    phase = "test" if test else "train" if pl_module.training else "val"

    the_metric = 0

    if (pl_module.hparams.m3ae_config["get_recall_metric"] and not pl_module.training) \
            or (test and pl_module.hparams.m3ae_config["loss_names"]["irtr"] >= 1):
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.log(f"{phase}/recalls/ir_r1", ir_r1)
        pl_module.log(f"{phase}/recalls/ir_r5", ir_r5)
        pl_module.log(f"{phase}/recalls/ir_r10", ir_r10)
        pl_module.log(f"{phase}/recalls/tr_r1", tr_r1)
        pl_module.log(f"{phase}/recalls/tr_r5", tr_r5)
        pl_module.log(f"{phase}/recalls/tr_r10", tr_r10)
        pl_module.logger.experiment[0].add_scalar("recalls/ir_r1", ir_r1, pl_module.global_step)
        pl_module.logger.experiment[0].add_scalar("recalls/ir_r5", ir_r5, pl_module.global_step)
        pl_module.logger.experiment[0].add_scalar("recalls/ir_r10", ir_r10, pl_module.global_step)
        pl_module.logger.experiment[0].add_scalar("recalls/tr_r1", tr_r1, pl_module.global_step)
        pl_module.logger.experiment[0].add_scalar("recalls/tr_r5", tr_r5, pl_module.global_step)
        pl_module.logger.experiment[0].add_scalar("recalls/tr_r10", tr_r10, pl_module.global_step)
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.m3ae_config["loss_names"].items():
        if v <= 0:
            continue
        value = 0
        if loss_name == "vqa":
            print(pl_module)
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            pl_module.log(f"{loss_name}/{phase}/score_best_epoch",
                          getattr(pl_module, f"{phase}_{loss_name}_score").get_best_score())
            pl_module.log(f"{loss_name}/{phase}/close_score_best_epoch",
                          getattr(pl_module, f"{phase}_{loss_name}_score").get_best_close_score())
            pl_module.log(f"{loss_name}/{phase}/open_score_best_epoch",
                          getattr(pl_module, f"{phase}_{loss_name}_score").get_best_open_score())
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()

            pl_module.log(f"{loss_name}/{phase}/loss_epoch", getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            # Log additional metrics: ROUGE1, ROUGE2, BLEU 
            metrics = ["rouge1", "rouge2", "bleu_score"]
            for metric in metrics:
                metric_value = getattr(pl_module, f"{phase}_{loss_name}_{metric}").compute()
                #pl_module.log(f"{loss_name}/{phase}/{metric}", metric_value)
                pl_module.log(f"{phase}/{metric}", metric_value)
                getattr(pl_module, f"{phase}_{loss_name}_{metric}").reset()            

        elif loss_name == "cls":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "irtr":
            value = getattr(pl_module, f"{phase}_irtr_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/irtr_loss_epoch", value)
            getattr(pl_module, f"{phase}_irtr_loss").reset()
            value = -value

        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "mim":
            value = -getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", - value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "mlm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        else:
            raise ValueError

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)

def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()

def set_task(pl_module):
    pl_module.current_tasks = [k for k, v in pl_module.hparams.m3ae_config["loss_names"].items() if v > 0]
    return

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def set_schedule(pl_module):
    """
    Configure optimizer and scheduler for M3AE-T5 module.

    Args:
        pl_module: PyTorch Lightning module with T5 and M3AE.

    Returns:
        Optimizer and scheduler.
    """
    lr = pl_module.hparams.m3ae_config["learning_rate"]
    wd = pl_module.hparams.m3ae_config["weight_decay"]
    lr_multiplier_head = pl_module.hparams.m3ae_config.get("lr_multiplier_head", 1.0)
    lr_multiplier_t5 = pl_module.hparams.m3ae_config.get("lr_multiplier_t5", 1.0)
    end_lr = pl_module.hparams.m3ae_config["end_lr"]
    decay_power = pl_module.hparams.m3ae_config["decay_power"]
    optim_type = pl_module.hparams.m3ae_config["optim_type"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
    ]
    head_names = ["mlm_head", "mim_head", "itm_head", "vqa_head", "cls_head", "irtr_head"]
    multi_modal_names = ["multi_modal"]
    t5_names = ["t5"]  # Parameters specific to T5.

    optimizer_grouped_parameters = [
        # Base parameters (excluding heads, multi-modal layers, T5).
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in multi_modal_names)
                and not any(tt in n for tt in t5_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in multi_modal_names)
                and not any(tt in n for tt in t5_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        # Head parameters (higher learning rate).
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_multiplier_head,
        },
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_multiplier_head,
        },
        # Multi-modal parameters.
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(ht in n for ht in multi_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_multiplier_head,
        },
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(ht in n for ht in multi_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_multiplier_head,
        },
        # T5-specific parameters (custom learning rate).
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(tt in n for tt in t5_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_multiplier_t5,
        },
        {
            "params": [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(tt in n for tt in t5_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_multiplier_t5,
        },
    ]

    # Optimizer selection.
    if optim_type == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    # Scheduler configuration.
    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.m3ae_config["warmup_steps"]
    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}
    return [optimizer], [sched]



