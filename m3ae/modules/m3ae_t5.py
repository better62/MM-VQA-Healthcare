import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F

from m3ae.modules import M3AETransformerSS
from m3ae.modules import m3ae_t5_utils

class T5VQA(pl.LightningModule):
    def __init__(self, config, max_answer_length=80, freeze_m3ae=True, freeze_t5_layers=True, projection_downsample_factor=4):
        super().__init__()
        self.save_hyperparameters()
        self.projection_downsample_factor = projection_downsample_factor

        # Initialize M3AE model
        self.m3ae = M3AETransformerSS(config)

        # Freeze M3AE if specified
        if freeze_m3ae:
            for param in self.m3ae.parameters():
                param.requires_grad = False
        
        # Initialize T5 with generation capabilities
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-small')

        # Freeze T5 layers if specified
        if freeze_t5_layers:
            for param in self.t5.parameters():
                param.requires_grad = False
        
        # Projection layer for M3AE features
        self.feature_projection = nn.Linear(
            config["hidden_size"] * 2,  # M3AE outputs concatenated features
            self.t5.config.hidden_size // self.projection_downsample_factor
        )
        
        self.max_answer_length = max_answer_length

    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze the top N layers of T5 encoder and decoder"""
        for param in self.t5.parameters():
            param.requires_grad = False

        # Unfreeze top N layers of decoder
        for i in range(len(self.t5.decoder.block) - num_layers, len(self.t5.decoder.block)):
            # Self attention
            for param in self.t5.decoder.block[i].layer[0].parameters():
                param.requires_grad = True
            # Cross attention
            for param in self.t5.decoder.block[i].layer[1].parameters():
                param.requires_grad = True
        
        # To see which layers are frozen
        # print("Called unfreeze_layers(" + str(num_layers) + ") on t5")
        # for param in self.t5.parameters():
        #     print(param.requires_grad)

    def create_positional_encoding(self, seq_length, hidden_size):
        """Create absolute positional encoding for T5"""
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(seq_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(self.device)

    def prepare_inputs(self, batch):

        # get multi_modal_cls_feats
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)
        multi_modal_features = m3ae_output["multi_modal_cls_feats"] # 1 x 1536
        
        # Prepare context and question prefixes
        context_prefix = "context:"
        question_prefix = "question:"
        
        # Tokenize prefixes
        tokenized_context_prefix = self.tokenizer(context_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        tokenized_question_prefix = self.tokenizer(question_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # Process each question individually
        t5_inputs = []
        for i, question in enumerate(batch["text"]):
            # Get the corresponding multi-modal feature for this question
            projected_feature = self.feature_projection(multi_modal_features[i:i+1])
            
            # Tokenize the current question
            tokenized_question = self.tokenizer(
                question, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.hparams.config["max_text_len"]
            ).input_ids.to(self.device)
            
            # Concatenate inputs in T5's expected format
            t5_input = torch.cat([
                tokenized_context_prefix, 
                projected_feature, 
                tokenized_question_prefix, 
                tokenized_question
            ], dim=1)
            
            # Add positional encoding
            seq_length = t5_input.shape[1]
            positional_encoding = self.create_positional_encoding(seq_length, self.t5.config.hidden_size)
            t5_input_with_pe = t5_input + positional_encoding
            
            t5_inputs.append(t5_input_with_pe)
    
        # Stack the inputs if needed
        t5_inputs = torch.stack(t5_inputs)
        
        return t5_inputs

    def forward(self, batch, labels=None):
        # Prepare encoder inputs with combined visual and textual features
        model_inputs = self.prepare_inputs(batch)

        encoder_outputs = self.t5.encoder(inputs_embeds=model_inputs) # batch size x 512
        
        if labels is not None:
            # Training mode
            label_tokens = self.tokenizer(
                labels,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Forward pass through T5 with labels for training
            outputs = self.t5(
                labels=label_tokens,
                encoder_outputs=encoder_outputs,
                attention_mask=model_inputs["attention_mask"],
                return_dict=True
            )
        else:
            # Inference mode
            outputs = self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=model_inputs["attention_mask"],
                max_length=self.max_answer_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        return outputs

    def generate_answer(self, batch):
        outputs = self(batch)
        
        # Decode generated sequences
        answers = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return answers

    def training_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self, test=True)

    def training_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self, test=True)


    def configure_optimizers(self):
        # Collect parameters by groups
        pretrained_params = []
        new_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'feature_projection' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': 1e-5},
            {'params': new_params, 'lr': 5e-5}
        ])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }
