from tsflow.module.vqvae import VQVAE
from gluonts.dataset.repository.datasets import get_dataset
from lightning import Trainer, LightningModule
import torch

class TestVQVAE(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VQVAE(
            in_channels=1,
            embedding_dim=64,
            num_embeddings=512,
            commitment_cost=0.25,
            decay=0.99,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        recon, vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings, data_recon = self.model.shared_eval(batch, None, 'train')
        loss = torch.mean((batch - recon) ** 2) + vq_loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers(lr=1e-3)


def test_vqvae_training_step():
    dataset = get_dataset("electricity")
    train_data = dataset.train
    
    import pdb; pdb.set_trace()

    model = TestVQVAE()
    trainer = Trainer(max_epochs=1, limit_train_batches=5)
    trainer.fit(model, train_dataloaders=train_data)

if __name__ == "__main__":
    test_vqvae_training_step()