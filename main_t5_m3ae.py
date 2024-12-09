import copy
import os
import resource
import gc
import torch
import wandb

import pytorch_lightning as pl 

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS
from m3ae.modules import T5VQA_MMEncoderInput

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data modules
    # update dist param to be optional
    #dm = MTDataModule(_config, dist=True)
    #dm = MTDataModule(_config, dist=False)  
    dm = MTDataModule(_config, dist=_config["use_ddp"])  


    # Module
    model = T5VQA_TextEncoderInput(_config)
    model.unfreeze_top_layers(num_encoder_layers=_config["unfreeze_num_encoder_layers"],num_decoder_layers=_config["unfreeze_num_decoder_layers"]) # Unfreeze the top 1 layers of DistilBERT

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)

    # Define project name here !!
    wb_logger = pl.loggers.WandbLogger(project="VQA-RAD-T5", name=run_name)
    loggers = [tb_logger, wb_logger]

    # Callbackttg
    '''
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/vqa/loss",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )
    '''
    # save every checkpoint per epoch
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        save_last=True,  
        save_weights_only=True if "finetune" in exp_name else False,
    )    

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else 1000


    gc.collect()
    torch.cuda.empty_cache()

    # Trainer 
    #os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # GPU device number -> maybe "gpus=[5]" is correct
    trainer = pl.Trainer(
        gpus=[_config["gpu_device_number"]], # GPU device number
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # distributed_backend="ddp",  
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"]
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
