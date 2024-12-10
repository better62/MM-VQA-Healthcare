import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F

from m3ae.modules import M3AETransformerSS
from m3ae.modules import m3ae_t5_utils
from m3ae.modules import objectives

class T5VQA_MMEncoderInput(pl.LightningModule):
    def __init__(self, m3ae_config, freeze_m3ae=True, freeze_t5_layers=True):
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

        if m3ae_config["load_path_t5"] != "":
            ckpt = torch.load(m3ae_config["load_path_t5"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        # Freeze T5 layers if specified
        if freeze_t5_layers:
            for param in self.t5.parameters():
                param.requires_grad = False
        
        # Projection layer for M3AE features
        self.feature_projection = nn.Linear(
            m3ae_config["hidden_size"] * 2,  # M3AE outputs concatenated features
            self.t5.config.hidden_size
        )
        
        self.max_answer_length = m3ae_config["t5_max_length"]
        m3ae_t5_utils.set_metrics(self)
        self.current_tasks = list()

        # Load checkpoint if path is provided
        if m3ae_config["load_path"] != "":
            self.load_model_checkpoint(m3ae_config["load_path"])        

    def load_model_checkpoint(self, ckpt_path):
        # Load the checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        
        # Load M3AE weights with positional encoding adjustment if needed
        if "m3ae" in state_dict:
            if self.m3ae.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=self.m3ae.hparams.config["image_size"],
                                                     patch_size=self.m3ae.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict,
                                                          after=self.m3ae.hparams.config["image_size"])
            self.m3ae.load_state_dict(state_dict, strict=False)
        
        # Load T5 weights
        t5_state_dict = {k[len("t5."):]: v for k, v in state_dict.items() if k.startswith("t5.")}
        self.t5.load_state_dict(t5_state_dict, strict=False)

        print("Checkpoint loaded successfully!")
    
    def projection_layer(self, input_dim, output_dim=512):
        linear_layer = nn.Linear(input_dim, output_dim).to(self.device)
        return linear_layer
    
    def unfreeze_top_layers(self, num_encoder_layers=2, num_decoder_layers=2):
        # Freeze all T5 layers by default
        for param in self.t5.parameters():
            param.requires_grad = False

        # Unfreeze top N encoder layers
        for i in range(len(self.t5.encoder.block) - num_encoder_layers, len(self.t5.encoder.block)):
            for param in self.t5.encoder.block[i].parameters():
                param.requires_grad = True

        # Unfreeze top N decoder layers
        for i in range(len(self.t5.decoder.block) - num_decoder_layers, len(self.t5.decoder.block)):
            # Self attention
            for param in self.t5.decoder.block[i].layer[0].parameters():
                param.requires_grad = True
            # Cross attention
            for param in self.t5.decoder.block[i].layer[1].parameters():
                param.requires_grad = True
        
        #print(f"Unfroze {num_encoder_layers} encoder layers and {num_decoder_layers} decoder layers.")

    def prepare_inputs(self, batch, include_cls_feats, include_imagetext_feats, mm_feats_width):
        # Get multi-modal features from M3AE
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)

        # Extract multi-modal, image, and text features
        if include_cls_feats:
            multi_modal_cls_feats = m3ae_output["multi_modal_cls_feats"]  # [batch_size, 1536]
        if include_imagetext_feats:
            multi_modal_image_feats = m3ae_output["multi_modal_image_feats"]  # [batch_size, seq_len, 768]
            multi_modal_text_feats = m3ae_output["multi_modal_text_feats"]  # [batch_size, seq_len, 768]
        
        batch_inputs = []
        attention_masks = []
        
        for i, question in enumerate(batch["text"]):
            
            # 1. Handle question prefix
            question_prefix = "question:"
            question_prefix_tokens = self.tokenizer(
                question_prefix,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            question_prefix_embeddings = self.t5.shared(question_prefix_tokens)  # 1 x prefix_len x 512
            
            # 2. Project multi-modal CLS features
            if include_cls_feats:
                cls_feature = multi_modal_cls_feats[i:i+1]  # [1, 1536]
                cls_proj_layer = self.projection_layer(cls_feature.shape[1])
                projected_cls_feature = cls_proj_layer(cls_feature)  # [1, 512]
                projected_cls_feature = projected_cls_feature.unsqueeze(1)  # [1, 1, 512]
            else:
                projected_cls_feature = torch.zeros((1, 0, 512)).to(self.device)

            # 3. Project image and text features
            if include_imagetext_feats:
                seq_len_proj = self.projection_layer(multi_modal_image_feats[i].shape[0], mm_feats_width // 2)  # seq_len -> mm_feats_width // 2
                projected_image_features = seq_len_proj(multi_modal_image_feats[i].T).T  # [mm_feats_width // 2, 768]
                embed_dim_proj = self.projection_layer(projected_image_features.shape[1])  # 768 -> 512
                projected_image_features = embed_dim_proj(projected_image_features)  # [mm_feats_width // 2, 512]
                projected_image_features = projected_image_features.unsqueeze(0)  # [1, mm_feats_width // 2, 512]

                seq_len_proj = self.projection_layer(multi_modal_text_feats[i].shape[0], mm_feats_width // 2)  # [seq_len -> mm_feats_width // 2]
                projected_text_features = seq_len_proj(multi_modal_text_feats[i].T).T  # [mm_feats_width // 2, 768]
                embed_dim_proj = self.projection_layer(projected_text_features.shape[1])  # 768 -> 512
                projected_text_features = embed_dim_proj(projected_text_features)  # [mm_feats_width // 2, 512]
                projected_text_features = projected_text_features.unsqueeze(0)  # [1, mm_feats_width // 2, 512]
            else:
                projected_image_features = torch.zeros((1, 0, 512)).to(self.device)
                projected_text_features = torch.zeros((1, 0, 512)).to(self.device)
            
            # 5. Combine question prefix with m3ae features
            combined_embeddings = torch.cat([
                question_prefix_embeddings,   # 1 x prefix_len x 512
                projected_cls_feature,       # 1 x 1 x 512
                projected_image_features,           # 1 x (mm_feats_width // 2) x 512
                projected_text_features,            # 1 x (mm_feats_width // 2) x 512
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


    def forward(self, batch, test=False):
        # Prepare encoder inputs with combined visual and textual features
        ret = dict()
        
        model_inputs = self.prepare_inputs(batch, 
                                           self.hparams.m3ae_config["mm_encoder_inputs_include_cls_feats"], 
                                           self.hparams.m3ae_config["mm_encoder_inputs_include_imagetext_feats"], 
                                           self.hparams.m3ae_config["mm_encoder_inputs_mm_feats_width"])

        encoder_outputs = self.t5.encoder(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"]
        )

        if len(self.current_tasks) == 0 or test:
            # Inference/test mode - generate text
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
            
            # Decode the generated outputs to text
            generated_texts = self.tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )
            generated_texts_ = [[item] for item in generated_texts]
            #ret.update({'outputs': generated_texts_})
            return ret

        else:
            # Training mode
            labels = batch["vqa_answer"]  # Assuming these are text answers
            flattened_labels = [label[0] for label in labels]
            # Tokenize the labels
            label_tokens = self.tokenizer(
                flattened_labels,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            #print(' : ',labels) # 8x1 [['no'], ['no'], ['abdomen'], ['yes'], ['right upper lobe'], ['yes'], ['no'], ['no']]
            #print('flattened_labels: ',flattened_labels) # ['no', 'no', 'abdomen', 'yes', 'right upper lobe', 'yes', 'no', 'no']
            #print('label_tokens: ',label_tokens) # 8x6 shape, contains integer (one answer -> 1x6)
            # Get model outputs
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=label_tokens,
                return_dict=True
            )
            #print('outputs shape: ',outputs.logits.shape) # torch.Size([8, 6, 32128])
            #print('outputs: ',outputs[:2]) # torch.Size([8, 6, 32128]) prob values. 32128== T5's vocab size
            # For metrics, we need to generate text even during training
            with torch.no_grad():
                generated_outputs = self.t5.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=self.max_answer_length,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                #print("Generated sequences shape:", generated_outputs.sequences.shape) # torch.Size([8, 80]) 80: T5 generation max_length
                # Decode generated text
                #print('generated_outputs: ',generated_outputs,'\n\n')
                # BeamSearchEncoderDecoderOutput(sequences=tensor([[ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                # 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                # 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                # 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                # 0,  0,  0,  0,  0,  0,  0,  0],
                # [ 0,  3,  6,  3,  6,  3, 11,  3,  6,  3,  6,  3,  6,  3, 11,  3,  6,  3,
                # 6,  3,  6,  3, 11,  3,  6,  3,  6,  3,  6,  3, 11,  3,  6,  3,  6,  3,
                # 11,  3,  6,  3,  6,  3, 11,  3,  6,  3,  6,  3,  6,  3, 11,  3,  6,  3,
                # 6,  3,  6,  3, 11,  3,  6,  3,  6,  3,  6,  3,  3,  3,  3,  3,  3,  3,
                # 3,  3,  3,  3,  3,  3,  3,  3],
                generated_texts = self.tokenizer.batch_decode(
                    generated_outputs.sequences, 
                    skip_special_tokens=True
                )
                # for i in range(len(generated_outputs.sequences)):
                    # print("Generated sequences:", generated_outputs.sequences[i]) 
                    # print('generated_texts: ',generated_texts[i])
                #print('generated_texts: ',len(generated_texts)) # 8
                #generated_texts:  ['', ',, a, e,,, s,,,,,,,   ,,,, s,,, ,, ,, e, , , ,  ,', "s '' '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''", ',', 'and        ,                                ,', 's. not not a s... s. ssmanmanman s dd ss. s.. s dd.. n  not not... s.. s s. s.... s', '', '']
                generated_texts_ = [[item] for item in generated_texts]
            
            ret.update(objectives.compute_vqa(
                self, 
                batch, 
                test=test, 
                outputs=generated_texts_,  # Pass decoded text
                loss=outputs.loss,
                labels=labels  # Pass original text labels
            ))
            
            return ret
        
        # if labels is not None:
        #     # For training, tokenize labels separately
        #     label_tokens = self.tokenizer(
        #         labels,
        #         padding=True,
        #         truncation=True,
        #         return_tensors="pt"
        #     ).input_ids.to(self.device)
            
        #     outputs = self.t5(
        #         encoder_outputs=encoder_outputs,
        #         labels=label_tokens,
        #         return_dict=True
        #     )

        #     loss = outputs.loss
        # else:
        #     # Inference mode
        #     outputs = self.t5.generate(
        #         encoder_outputs=encoder_outputs,
        #         max_length=self.max_answer_length,
        #         num_beams=4,
        #         early_stopping=True,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         eos_token_id=self.tokenizer.eos_token_id,
        #         return_dict_in_generate=True,
        #         output_scores=True
        #     )
            
        

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
        total_loss = sum([v * self.hparams.m3ae_config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        return {'loss': total_loss}

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
        return m3ae_t5_utils.set_schedule(self)
        # Collect parameters by groups
        # pretrained_params = []
        # new_params = []
        
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         if 'feature_projection' in name:
        #             new_params.append(param)
        #         else:
        #             pretrained_params.append(param)
        
        # optimizer = torch.optim.AdamW([
        #     {'params': pretrained_params, 'lr': 1e-5},
        #     {'params': new_params, 'lr': 5e-5}
        # ])
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=10, eta_min=1e-6
        # )
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "train_loss"
        #     }
        # }