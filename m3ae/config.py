from sacred import Experiment

ex = Experiment("METER", save_git_info=False)

def _loss_names(d):
    ret = {
        "mlm": 0,
        "mim": 0,
        "itm": 0,
        "vqa": 0,
        "cls": 0,
        "irtr": 0
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "meter"
    seed = 0
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqa_label_size = 3129
    mlc_label_size = 14
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # MIM decoder Setting
    mim_prob = 0.75
    mim_decoder_hidden_size = 384
    mim_decoder_num_layers = 4
    mim_decoder_num_heads = 6
    norm_pix_loss = True
    mim_layer = -1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = -1
    warmup_steps = 10000
    end_lr = 0
    lr_multiplier_head = 5  # multiply lr for prediction heads
    lr_multiplier_multi_modal = 5  # multiply lr for the multi-modal module

    # Encoder Setting
    mm_encoder_inputs_include_cls_feats = True
    mm_encoder_inputs_include_imagetext_feats = False
    mm_encoder_inputs_mm_feats_width = 0

    # T5 Model Setting
    t5_model_name = "t5-small" 
    t5_max_length = 25  
    t5_generation = True 

    # Unfreeze layer number Setting
    unfreeze_num_encoder_layers = 2
    unfreeze_num_decoder_layers = 2

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    default_root_dir = "checkpoints"

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0
    use_ddp = False
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    gpu_device_number = 0

    # MELINDA SETTING
    label_column_name = ""
    melinda_label_size = {"i_meth": 85, "p_meth": 45, "i_meth_label": 15, "p_meth_label": 7}

    #WANDB setting
    api_key = "20be045acce9a973c8a3780aaba86927c1fc5b83"

@ex.named_config
def task_pretrain_m3ae():
    exp_name = "task_pretrain_m3ae"
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mim": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 64
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_multiplier_head = 5
    lr_multiplier_multi_modal = 5
    num_top_layer = 6
    hidden_size = 768
    num_heads = 12

    precision = 16
    mim_layer = 3

@ex.named_config
def task_finetune_vqa_ehr_xqa():
    exp_name = "task_finetune_vqa_ehr_xqa"
    datasets = ["vqa_ehr_xqa"]
    loss_names = _loss_names({"vqa": 1}) 
    batch_size = 64
    max_epoch = 50
    max_steps = 1000
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576
    vqa_label_size = 700
    max_text_len = 32


@ex.named_config
def task_finetune_vqa_vqa_rad():
    exp_name = "task_finetune_vqa_vqa_rad"
    datasets = ["vqa_vqa_rad"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 64
    max_epoch = 20
    max_steps = 1000
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_multiplier_head = 100
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 16
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576
    vqa_label_size = 498
    max_text_len = 32


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

# vision encoder
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024


@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024


@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536


@ex.named_config
def clip32():
    vit = 'ViT-B/32'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


@ex.named_config
def clip16():
    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


# text encoder
@ex.named_config
def text_roberta():
    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


@ex.named_config
def text_roberta_large():
    tokenizer = "roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024


# random augmentation
@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]


@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]


@ex.named_config
def clip_resizedcrop():
    train_transform_keys = ["clip_resizedcrop"]
