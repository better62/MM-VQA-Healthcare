import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
# from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

from m3ae.modules import M3AETransformerSS


class DistilBERTVQA(pl.LightningModule):
    def __init__(self, m3ae_config, max_answer_length=80, freeze_m3ae=True, freeze_distilbert_layers=True):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize M3AE model
        self.m3ae = M3AETransformerSS(m3ae_config)

        # Freeze M3AE if specified
        if freeze_m3ae:
            for param in self.m3ae.parameters():
                param.requires_grad = False
        
        # Initialize DistilBERT
        # local_model_path = os.path.abspath("downloaded/distilbert-base-uncased")
        # self.tokenizer = DistilBertTokenizer.from_pretrained(local_model_path, local_files_only=True)
        # self.distilbert = DistilBertModel.from_pretrained(local_model_path, local_files_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained("nlpie/clinical-distilbert")
        self.distilbert = AutoModelForMaskedLM.from_pretrained("nlpie/clinical-distilbert")

        # Freeze DistilBERT layers if specified
        if freeze_distilbert_layers:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        self.max_answer_length = max_answer_length
        m3ae_t5_utils.set_metrics(self)
        self.current_tasks = list()

    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze the top N layers of DistilBERT"""
        for param in self.distilbert.parameters():
            param.requires_grad = False
            
        # DistilBERT has 6 layers, so we'll unfreeze from the top
        for i in range(6 - num_layers, 6):
            for param in self.distilbert.transformer.layer[i].parameters():
                param.requires_grad = True

    def forward(self, batch, answer_tokens=None):
        # Extract multi-modal features using M3AE
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)

        multi_modal_cls_feats = torch.cat([
            m3ae_output["multi_modal_image_feats"], 
            m3ae_output["multi_modal_text_feats"]
        ], dim=1)
        
        # Get DistilBERT embeddings for the question
        question_tokens = self.tokenizer(
            batch["text_ids"], 
            padding=True, 
            return_tensors="pt"
        ).to(multi_modal_features.device)
        
        question_embeddings = self.distilbert(**question_tokens).last_hidden_state
        
        # During training, include partial answer for next token prediction
        if answer_tokens is not None:
            answer_embeddings = self.distilbert(
                input_ids=answer_tokens
            ).last_hidden_state
            
            # Combine features, question, and partial answer
            combined_features = torch.cat([
                projected_features.unsqueeze(1),
                question_embeddings,
                answer_embeddings
            ], dim=1)
        else:
            # For inference, only use features and question
            combined_features = torch.cat([
                projected_features.unsqueeze(1),
                question_embeddings
            ], dim=1)
        
        # Get token predictions
        token_logits = self.token_predictor(combined_features)
        
        return token_logits

    def generate_answer(self, batch, num_beams=4):
        device = next(self.parameters()).device
        batch_size = len(batch["text_ids"])
        
        # Initialize with start tokens
        current_sequences = torch.full(
            (batch_size * num_beams, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        beam_scores = torch.zeros(batch_size * num_beams, device=device)
        
        for step in range(self.max_answer_length):
            # Get predictions for next token
            logits = self(batch, current_sequences)[:, -1, :]
            next_token_scores = F.log_softmax(logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(1)
            
            # Reshape for beam search
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )
            
            # Get top-k next tokens and their scores
            next_scores, next_tokens = next_token_scores.topk(
                num_beams, dim=1, largest=True, sorted=True
            )
            
            # Update sequences
            next_sequences = torch.cat([
                current_sequences,
                next_tokens.view(-1, 1)
            ], dim=1)
            
            # Update beam scores
            beam_scores = next_scores.view(-1)
            current_sequences = next_sequences
            
            # Check for end of sequence
            if (current_sequences == self.tokenizer.eos_token_id).any(dim=-1).all():
                break
        
        # Get best sequence for each batch
        final_sequences = current_sequences.view(batch_size, num_beams, -1)
        best_sequences = final_sequences[:, 0, :]  # Take top beam
        
        # Decode answers
        answers = self.tokenizer.batch_decode(
            best_sequences,
            skip_special_tokens=True
        )
        
        return answers

    def training_step(self, batch, batch_idx):
        # Get target answers
        target_answers = batch["answers"]
        target_tokens = self.tokenizer(
            target_answers,
            padding=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # For each position, predict next token
        loss = 0
        for i in range(target_tokens.size(1) - 1):
            input_tokens = target_tokens[:, :i+1]
            logits = self(batch, input_tokens)
            
            # Calculate loss for next token prediction
            next_token_logits = logits[:, -1, :]
            next_token_targets = target_tokens[:, i+1]
            
            loss += F.cross_entropy(
                next_token_logits,
                next_token_targets,
                ignore_index=self.tokenizer.pad_token_id
            )
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Collect parameters by groups
        pretrained_params = []
        new_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'feature_projection' in name or 'token_predictor' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': 1e-5},  # Lower learning rate for pretrained layers
            {'params': new_params, 'lr': 5e-5}  # Higher learning rate for new layers
        ])
        
        # Optional: Add learning rate scheduler
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