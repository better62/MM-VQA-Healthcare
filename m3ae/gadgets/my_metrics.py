import numpy as np
import sklearn.metrics as sklm
import torch
from torchmetrics import Metric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizerFast
from collections import Counter

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros_like(target).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


#### Rouge1 score
class ROUGE1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("rouge1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_unigrams = Counter(pred.tolist())
            target_unigrams = Counter(target.tolist())
            overlap = sum((pred_unigrams & target_unigrams).values())  # 공통된 1-gram 개수
            possible_overlap = sum(target_unigrams.values())  # target의 총 1-gram 개수
            
            if possible_overlap > 0:
                self.rouge1 += overlap / possible_overlap
                self.total += 1

    def compute(self):
        return self.rouge1 / self.total if self.total > 0 else torch.tensor(0.0)

class ROUGE2Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("rouge2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_bigrams = Counter(zip(pred.tolist(), pred.tolist()[1:]))
            target_bigrams = Counter(zip(target.tolist(), target.tolist()[1:]))
            overlap = sum((pred_bigrams & target_bigrams).values())  # 공통된 2-gram 개수
            possible_overlap = sum(target_bigrams.values())  # target의 총 2-gram 개수
            
            if possible_overlap > 0:
                self.rouge2 += overlap / possible_overlap
                self.total += 1

    def compute(self):
        return self.rouge2 / self.total if self.total > 0 else torch.tensor(0.0)

        
#### Bleu score
class BLEUScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.smoothing = SmoothingFunction().method1  # Smoothing function 설정

    def update(self, predictions, references):
        for pred, ref in zip(predictions, references):
            pred_text = " ".join(map(str, pred.tolist())) if isinstance(pred, torch.Tensor) else pred
            ref_text = " ".join(map(str, ref.tolist())) if isinstance(ref, torch.Tensor) else ref

            # Smoothing function 적용
            bleu_score = sentence_bleu([ref_text.split()], pred_text.split(), smoothing_function=self.smoothing)
            self.score += bleu_score
            self.total += 1

    def compute(self):
        return self.score / self.total


class VQARADScore(VQAScore):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.best_score = 0 
        self.best_close_score = 0
        self.best_open_score = 0

    def update(self, logits, target, types=None):
        super().update(logits, target)
        close_scores = (types == 0).float() * self.score
        open_scores = (types == 1).float() * self.score

        self.close_score += close_scores.sum()
        self.close_total += close_scores.numel()
        self.open_score += open_scores.sum()
        self.open_total += open_scores.numel()

    def get_best_score(self):
        if (self.score / self.total) > self.best_score:
            self.best_score = self.compute()
            self.best_close_score = self.close_score / self.close_total if self.close_total != 0 else 0
            self.best_open_score = self.open_score / self.open_total if self.open_total != 0 else 0
        return self.best_score

    def get_best_close_score(self):
        return self.best_close_score

    def get_best_open_score(self):
        return self.best_open_score


class ROCScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )
        self.y_trues.append(target)
        self.y_scores.append(torch.sigmoid(logits))

    def compute(self):
        try:
            score = sklm.roc_auc_score(
                np.concatenate([y.cpu().numpy() for y in self.y_trues], axis=0),
                np.concatenate([y.cpu().numpy() for y in self.y_scores], axis=0)
            )
            return torch.tensor(score, device=self.y_trues[0].device)
        except ValueError:
            return torch.tensor(0.0, device=self.y_trues[0].device)


class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_preds", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )
        y_pred = (torch.sigmoid(logits) > 0.5).float()
        self.y_trues.append(target)
        self.y_preds.append(y_pred)

    def compute(self):
        try:
            score = sklm.f1_score(
                np.concatenate([y.cpu().numpy() for y in self.y_trues], axis=0),
                np.concatenate([y.cpu().numpy() for y in self.y_preds], axis=0)
            )
            return torch.tensor(score, device=self.y_trues[0].device)
        except ValueError:
            return torch.tensor(0.0, device=self.y_trues[0].device)
