import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F

current_dir = Path(__file__).parent.resolve()
allRank_dir = current_dir.parent / "allRank"
print(allRank_dir.as_posix())

sys.path.insert(1, (allRank_dir / "allrank").as_posix())
sys.path.append(allRank_dir.as_posix())
sys.path.append((allRank_dir / "allrank" / "models" / "losses").as_posix())

try:
    import neuralNDCG
except ImportError as e:
    print(e)
    raise ImportError(
        "Could not import neuralNDCG. " "Make sure allRank is cloned in the project root and the path is correct."
    )


def calculate_mrr(pred_batch: np.ndarray, label_batch: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a batch.

    Args:
        pred_batch (np.ndarray): shape (B, D), predicted scores for each document.
        label_batch (np.ndarray): shape (B, D), binary relevance labels (0 or 1).

    Returns:
        float: the MRR over the batch.
    """
    # Number of queries in the batch
    B, D = pred_batch.shape
    # For each query, get the indices that would sort scores descending
    # sorted_idx has shape (B, D)
    sorted_idx = np.argsort(-pred_batch, axis=1)

    # Initialize reciprocal ranks
    rranks = np.zeros(B, dtype=float)

    for i in range(B):
        # labels in the order of descending score
        rels_sorted = label_batch[i, sorted_idx[i]]
        # find the first relevant document
        hits = np.where(rels_sorted == 1)[0]
        if hits.size > 0:
            # rank is zero-based, so +1
            rank = hits[0] + 1
            rranks[i] = 1.0 / rank
        else:
            # if no relevant docs, reciprocal rank is 0
            rranks[i] = 0.0

    # Mean over the batch
    return rranks.mean()


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: [batch_size, seq_len] or [batch_size, seq_len, d_model]
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.pos_embedding(positions)
        return x + pos_embed


class RetrievalNet(pl.LightningModule):
    """
    Retrieval network using a ViT backbone, Transformer over model+patch tokens,
    combined BCE + neuralNDCG loss for ranking.
    """

    def __init__(
        self,
        vit: nn.Module,
        num_models: int,
        lr: float,
        alpha: float,
        num_layers: int,
        nheads: int,
    ):
        super().__init__()
        # save all except the actual vit module
        self.save_hyperparameters(ignore=["vit"])

        # vision transformer encoder (no classification head)
        self.img_encoder = vit
        # feature dimension
        D = self.img_encoder.num_features

        # learnable positional embeddings for sequence length: 1 (model token) + num_patches
        # ViT-base patch16 @224 → 14×14=196 patches → seq_len = 197
        seq_len = 1 + (self.img_encoder.patch_embed.num_patches)
        self.pos_embed = LearnablePositionalEmbedding(D, seq_len)

        # learnable model embeddings: each model id → D vector
        self.model_emb = nn.Embedding(num_models, D)

        # Transformer encoder stacking
        encoder_layer = TransformerEncoderLayer(
            d_model=D,
            nhead=nheads,
            dim_feedforward=4 * D,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # output head: map token embedding → score
        self.fc_out = nn.Linear(D, 1)

        self.sigmoid = nn.Sigmoid()

        # classification loss

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(num_models - 1))

        # losses
        self.stage_losses = {"train": [], "val": [], "test": []}
        self.stage_mrrs = {"train": [], "val": [], "test": []}
        self.epoch_losses = {"train": [], "val": [], "test": []}
        self.epoch_mrrs = {"train": [], "val": [], "test": []}

    def forward(self, img: torch.Tensor, model_ids: torch.LongTensor) -> torch.Tensor:
        """
        img: (B, C, H, W)
        model_ids: (B, M)
        returns logits: (B, M)
        """
        # ViT forward: returns (B, 1+patches, D)
        feat_seq = self.img_encoder.forward_features(img)  # (B, 197, D)
        # drop original CLS token; we'll use our own model token
        patch_feats = feat_seq[:, 1:, :]  # (B, P, D)
        B, P, D = patch_feats.shape

        # expand image patches per model
        M = model_ids.size(1)
        # repeat image patch sequence for each model in the batch
        img_seq = patch_feats.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, P, D)
        img_seq = img_seq.reshape(B * M, P, D)  # (B*M, P, D)

        # model token embeddings
        model_tok = self.model_emb(model_ids)  # (B, M, D)
        model_tok = model_tok.reshape(B * M, 1, D)  # (B*M, 1, D)

        # concatenate [model_token, patch_tokens]
        seq = torch.cat([model_tok, img_seq], dim=1)  # (B*M, 1+P, D)
        # add positional embeddings
        seq = self.pos_embed(seq)

        # transformer expects (seq_len, batch_size, dim)
        seq = seq.permute(1, 0, 2)  # (1+P, B*M, D)
        out = self.transformer(seq)  # (1+P, B*M, D)
        out = out.permute(1, 0, 2)  # (B*M, 1+P, D)

        # take the first token (model token) as representation
        model_out = out[:, 0, :]  # (B*M, D)
        logits = self.fc_out(model_out)  # (B*M, 1)
        # logits = self.sigmoid(logits)
        logits = logits.view(B, M)  # (B, M)
        return logits

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Combine cross-entropy on logits and neuralNDCG on softmax probabilities.
        labels: one-hot (B, M)
        """
        bce_loss = self.bce(logits, labels)
        ndcg_loss = neuralNDCG.neuralNDCG(self.sigmoid(logits), labels)

        # weighted sum
        return self.hparams.alpha * bce_loss + (1 - self.hparams.alpha) * ndcg_loss

    @torch.no_grad()
    def _calculate_mrr(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return calculate_mrr(logits.cpu().numpy(), labels.cpu().numpy())

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        Single step for train/val: compute logits and loss.
        Expects batch = (imgs, labels, model_ids).
        """
        imgs, labels, model_ids = batch
        logits = self(imgs, model_ids)
        loss = self._calculate_loss(logits, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        mrr = self._calculate_mrr(logits, labels)
        self.log(f"{stage}_mrr", mrr, prog_bar=True)
        self.stage_losses[stage].append(loss)
        self.stage_mrrs[stage].append(mrr)
        return loss

    def _on_epoch_end(self, stage: str) -> None:
        """
        Called at the end of each epoch to log average loss and MRR.
        """
        avg_loss = torch.stack(self.stage_losses[stage]).mean()
        avg_mrr = torch.tensor(self.stage_mrrs[stage]).mean()
        self.epoch_losses[stage].append(avg_loss)
        self.epoch_mrrs[stage].append(avg_mrr)
        self.stage_losses[stage] = []
        self.stage_mrrs[stage] = []

    def on_train_epoch_end(self):
        return self._on_epoch_end("train")

    # def on_validation_epoch_end(self):
    #     return self._on_epoch_end("val")

    def on_test_epoch_end(self):
        return self._on_epoch_end("test")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, "train")

    # def validation_step(self, batch, batch_idx: int) -> None:
    #     self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
