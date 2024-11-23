import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F

from m3ae.modules import M3AETransformerSS
from m3ae.modules import m3ae_t5_utils

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
        m3ae_t5_utils.set_metrics(self)

    def projection_layer(self, input_dim, output_dim=512):
        """Creates a Linear layer and projects input to output_dim."""
        linear_layer = nn.Linear(input_dim, output_dim).to(self.device)
        return linear_layer
    
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

    def prepare_inputs(self, batch):
        # Get multi-modal features from M3AE
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)

        # Extract multi-modal, image, and text features
        multi_modal_cls_feats = m3ae_output["multi_modal_cls_feats"]  # [batch_size, 1536]
        multi_modal_image_feats = m3ae_output["multi_modal_image_feats"]  # [batch_size, seq_len, 768]
        multi_modal_text_feats = m3ae_output["multi_modal_text_feats"]  # [batch_size, seq_len, 768]
        
        batch_inputs = []
        attention_masks = []
        
        for i, question in enumerate(batch["text"]):
            # 1. Handle context prefix
            context_prefix = "context:"
            context_tokens = self.tokenizer(
                context_prefix,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            context_embeddings = self.t5.shared(context_tokens)  # 1 x context_len x 512
            
            # 2. Project multi-modal CLS features
            cls_feature = multi_modal_cls_feats[i:i+1]  # [1, 1536]
            cls_proj_layer = self.projection_layer(cls_feature.shape[1])
            projected_cls_feature = cls_proj_layer(cls_feature)  # [1, 512]
            projected_cls_feature = projected_cls_feature.unsqueeze(1)  # [1, 1, 512]

            # 3. Average and project image features
            avg_image_features = torch.mean(multi_modal_image_feats[i], dim=0, keepdim=True)  # [1, 768]
            image_proj_layer = self.projection_layer(avg_image_features.shape[1])
            projected_image_features = image_proj_layer(avg_image_features)  # [1, 512]
            projected_image_features = projected_image_features.unsqueeze(0)  # [1, 1, 512]

            # 4. Average and project text features
            avg_text_features = torch.mean(multi_modal_text_feats[i], dim=0, keepdim=True)  # [1, 768]
            text_proj_layer = self.projection_layer(avg_text_features.shape[1])
            projected_text_features = text_proj_layer(avg_text_features)  # [1, 512]
            projected_text_features = projected_text_features.unsqueeze(0)  # [1, 1, 512]
            
            # 5. Handle question prefix
            question_prefix = "question:"
            question_prefix_tokens = self.tokenizer(
                question_prefix,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            question_prefix_embeddings = self.t5.shared(question_prefix_tokens)  # 1 x prefix_len x 512
            
            # 6. Handle actual question
            question_tokens = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=self.hparams.m3ae_config["max_text_len"]
            ).input_ids.to(self.device)
            question_embeddings = self.t5.shared(question_tokens)  # 1 x question_len x 512
            
            # 7. Combine all embeddings
            combined_embeddings = torch.cat([
                context_embeddings,           # 1 x context_len x 512
                projected_cls_feature,       # 1 x 1 x 512
                projected_image_features,           # 1 x 1 x 512
                projected_text_features,            # 1 x 1 x 512
                question_prefix_embeddings,   # 1 x prefix_len x 512
                question_embeddings           # 1 x question_len x 512
            ], dim=1)

            MAX_SEQ_LEN = 512
            # Pad or truncate to ensure length is 512
            seq_len = combined_embeddings.shape[1]
            if seq_len < MAX_SEQ_LEN:
                # Pad to 512 if shorter
                padding_size = MAX_SEQ_LEN - seq_len
                combined_embeddings = torch.nn.functional.pad(
                    combined_embeddings, 
                    (0, 0, 0, padding_size)  # Pad on the sequence dimension (dim=1)
                )
            elif seq_len > MAX_SEQ_LEN:
                # Truncate to 512 if longer
                combined_embeddings = combined_embeddings[:, :MAX_SEQ_LEN]
            
            # 8. Create attention mask for the entire sequence
            attention_mask = torch.ones(
                combined_embeddings.shape[1],
                dtype=torch.long,
                device=self.device
            )
            
            batch_inputs.append(combined_embeddings)
            attention_masks.append(attention_mask)
        
        # Stack all inputs and masks
        inputs_embeds = torch.cat(batch_inputs, dim=0)  # batch_size x total_seq_len x 512
        attention_mask = torch.stack(attention_masks)   # batch_size x total_seq_len
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask
        }


    def forward(self, batch, labels=None, test=False):
        # Prepare encoder inputs with combined visual and textual features
        model_inputs = self.prepare_inputs(batch)

        encoder_outputs = self.t5.encoder(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"]
        )
        
        if labels is not None:
            # For training, tokenize labels separately
            label_tokens = self.tokenizer(
                labels,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=label_tokens,
                return_dict=True
            )
        else:
            # Inference mode
            outputs = self.t5.generate(
                encoder_outputs=encoder_outputs,
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
