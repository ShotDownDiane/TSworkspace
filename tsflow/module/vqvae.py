import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
        super(Encoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 8:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = F.relu(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_B(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor):
        super(Decoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            # To get the correct shape back the kernel size has to be 5 not 4
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=5,
                                                    stride=3, padding=1)

            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_B = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 8:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 12:
            x = self._conv_1(inputs)
            x = self._residual_stack(x)

            x = self._conv_trans_2(x)
            x = F.relu(x)

            x = self._conv_trans_3(x)
            x = F.relu(x)

            x = self._conv_trans_4(x)

            return torch.squeeze(x)

        elif compression_factor == 16:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_B(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        """A straightforward vector quantizer.

        Accepts inputs of shape (B, C, L) where C is the embedding dim (channels).
        Returns:
          loss: commitment + quantization loss
          quantized: (B, C, L) tensor (with straight-through estimator)
          perplexity: scalar
          embedding_weight: embedding matrix (K, C)
          encoding_indices: (B, L) indices of selected embeddings
          encodings: (B*L, K) one-hot encodings
        """
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # use an embedding layer (keeps compatibility with previous code)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        nn.init.uniform_(self._embedding.weight, -1.0 / self._num_embeddings, 1.0 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Expected inputs: (B, C, L)
        if inputs.dim() != 3:
            raise ValueError("VectorQuantizer expects inputs with shape (B, C, L)")

        b, c, l = inputs.shape

        # flatten to (B*L, C)
        flat = inputs.permute(0, 2, 1).contiguous().view(-1, c)  # (B*L, C)

        # embedding weights (K, C)
        embed = self._embedding.weight  # (K, C)

        # compute distances (B*L, K)
        # dist = ||x||^2 + ||e||^2 - 2 x.e
        flat_sq = torch.sum(flat ** 2, dim=1, keepdim=True)  # (B*L, 1)
        embed_sq = torch.sum(embed ** 2, dim=1)  # (K,)
        distances = flat_sq + embed_sq.unsqueeze(0) - 2.0 * torch.matmul(flat, embed.t())

        # get encoding indices and one-hot encodings
        encoding_indices = torch.argmin(distances, dim=1)  # (B*L,)
        encodings = F.one_hot(encoding_indices, num_classes=self._num_embeddings).type(flat.dtype)  # (B*L, K)

        # quantize and unflatten to (B, C, L)
        quantized = torch.matmul(encodings, embed).view(b, l, c).permute(0, 2, 1).contiguous()

        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape encoding indices to (B, L)
        encoding_indices = encoding_indices.view(b, l)

        return loss, quantized, perplexity, embed, encoding_indices, encodings


class VQVAE(nn.Module):
    """A clearer, more standard VQ-VAE implementation.

    This class keeps the original `shared_eval` signature for compatibility but
    exposes `encode`, `decode` and `forward` helpers for clarity.
    """
    def __init__(self, vqvae_config):
        super().__init__()
        num_hiddens = vqvae_config['block_hidden_size']
        num_residual_layers = vqvae_config['num_residual_layers']
        num_residual_hiddens = vqvae_config['res_hidden_size']
        embedding_dim = vqvae_config['embedding_dim']
        num_embeddings = vqvae_config['num_embeddings']
        commitment_cost = vqvae_config['commitment_cost']
        self.compression_factor = vqvae_config['compression_factor']

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, self.compression_factor)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, self.compression_factor)

    def encode(self, x):
        """Encode input time series to latent pre-quantized space.

        Input x: (B, L) or (B, 1, L). Returns (B, C, L') where C == embedding_dim.
        """
        return self.encoder(x, self.compression_factor)

    def decode(self, z):
        """Decode quantized latents back to reconstruction shape (B, L)."""
        return self.decoder(z, self.compression_factor)

    def forward(self, x):
        """Run full forward: encode -> quantize -> decode.

        Returns: recon, vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings
        """
        z = self.encode(x)
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
        recon = self.decode(quantized)
        return recon, vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        """Compatibility wrapper used by the rest of the codebase.

        mode: 'train' | 'val' | 'test'
        Returns same tuple as the old implementation.
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else batch.device
        loss = None
        vq_loss = None
        recon_error = None
        data_recon = None
        perplexity = None
        embedding_weight = None
        encoding_indices = None
        encodings = None
        
        if mode == 'train':
            assert optimizer is not None, "optimizer must be provided in train mode"
            optimizer.zero_grad()
            z = self.encode(batch)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decode(quantized)
            recon_error = F.mse_loss(data_recon, batch)
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

        elif mode in ('val', 'test'):
            with torch.no_grad():
                z = self.encode(batch)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
                data_recon = self.decode(quantized)
                recon_error = F.mse_loss(data_recon, batch)
                loss = recon_error + vq_loss

        # logging if provided
        if comet_logger is not None:
            # some loggers expect scalar floats
            try:
                comet_logger.log_metric(f'{mode}_vqvae_loss_each_batch', float(loss.item()))
                comet_logger.log_metric(f'{mode}_vqvae_vq_loss_each_batch', float(vq_loss.item()))
                comet_logger.log_metric(f'{mode}_vqvae_recon_loss_each_batch', float(recon_error.item()))
                comet_logger.log_metric(f'{mode}_vqvae_perplexity_each_batch', float(perplexity.item()))
            except Exception:
                # be permissive: don't crash training if logger fails
                pass

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings

    def configure_optimizers(self, lr=1e-3):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # adds weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer


