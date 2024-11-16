import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F

from m3ae.modules import M3AETransformerSS
https://github.com/pulls
class T5VQA(pl.LightningModule):
    def __init__(self, m3ae_config, max_answer_length=80, freeze_m3ae=True, freeze_t5_layers=True):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize M3AE model
        self.m3ae = M3AETransformerSS(m3ae_config)

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
            m3ae_config["hidden_size"] * 2,  # M3AE outputs concatenated features
            self.t5.config.hidden_size
        )
        
        self.max_answer_length = max_answer_length

    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze the top N layers of T5 encoder and decoder"""
        for param in self.t5.parameters():
            param.requires_grad = False
        
        # Unfreeze top N layers of encoder
        for i in range(len(self.t5.encoder.block) - num_layers, len(self.t5.encoder.block)):
            for param in self.t5.encoder.block[i].layer[0].parameters():
                param.requires_grad = True

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

    def prepare_inputs(self, batch):
        # Extract multi-modal features using M3AE
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)
        multi_modal_features = m3ae_output["multi_modal_cls_feats"]
        
        # Project features to T5 dimension
        projected_features = self.feature_projection(multi_modal_features)
        
        # Tokenize questions
        question_tokens = self.tokenizer(
            batch["text_ids"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(multi_modal_features.device)
        
        # Get T5 question embeddings
        encoder_outputs = self.t5.encoder(
            **question_tokens,
            return_dict=True
        )
        
        # Combine projected features with question embeddings
        combined_hidden_states = torch.cat([
            projected_features.unsqueeze(1),
            encoder_outputs.last_hidden_state
        ], dim=1)
        
        return {
            "encoder_outputs": (combined_hidden_states,),
            "attention_mask": torch.ones(
                combined_hidden_states.shape[:2],
                device=combined_hidden_states.device
            )
        }

    def forward(self, batch, labels=None):
        # Prepare encoder inputs with combined visual and textual features
        model_inputs = self.prepare_inputs(batch)
        
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
                encoder_outputs=model_inputs["encoder_outputs"],
                attention_mask=model_inputs["attention_mask"],
                return_dict=True
            )
        else:
            # Inference mode
            outputs = self.t5.generate(
                encoder_outputs=model_inputs["encoder_outputs"],
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
        outputs = self(batch, labels=batch["answers"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

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
