# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from recommenders.models.sasrec.model import SASREC, Encoder, LayerNormalization


class SSEPT(SASREC):
    """
    SSE-PT Model

    :Citation:

    Wu L., Li S., Hsieh C-J., Sharpnack J., SSE-PT: Sequential Recommendation
    Via Personalized Transformer, RecSys, 2020.
    TF 1.x codebase: https://github.com/SSE-PT/SSE-PT
    TF 2.x codebase (SASREc): https://github.com/nnkkmto/SASRec-tf2
    """

    def __init__(self, **kwargs):
        """Model initialization.

        Args:
            item_num (int): Number of items in the dataset.
            seq_max_len (int): Maximum number of items in user history.
            num_blocks (int): Number of Transformer blocks to be used.
            embedding_dim (int): Item embedding dimension.
            attention_dim (int): Transformer attention dimension.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout rate.
            l2_reg (float): Coefficient of the L2 regularization.
            num_neg_test (int): Number of negative examples used in testing.
            user_num (int): Number of users in the dataset.
            user_embedding_dim (int): User embedding dimension.
            item_embedding_dim (int): Item embedding dimension.
        """
        super().__init__(**kwargs)

        self.user_num = kwargs.get("user_num", None)  # New
        self.conv_dims = kwargs.get("conv_dims", [200, 200])  # modified
        self.user_embedding_dim = kwargs.get(
            "user_embedding_dim", self.embedding_dim
        )  # extra
        self.item_embedding_dim = kwargs.get("item_embedding_dim", self.embedding_dim)
        self.hidden_units = self.item_embedding_dim + self.user_embedding_dim

        # New, user embedding
        self.user_embedding_layer = nn.Embedding(
            self.user_num + 1,
            self.user_embedding_dim,
            padding_idx=0,
        )

        self.positional_embedding_layer = nn.Embedding(
            self.seq_max_len,
            self.user_embedding_dim + self.item_embedding_dim,  # difference
        )

        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.hidden_units,
            self.hidden_units,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.hidden_units, 1e-08
        )

    def forward(self, x, training=True):
        """Model forward pass.

        Args:
            x (dict): Input dictionary containing 'users', 'input_seq', 'positive', 'negative'.
            training (bool): Training mode flag.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor:
            - Logits of the positive examples.
            - Logits of the negative examples.
            - Mask for nonzero targets
        """
        users = x["users"]
        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = (input_seq != 0).float().unsqueeze(-1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # User Encoding
        u_latent = self.user_embedding_layer(users)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)

        # replicate the user embedding for all the items
        u_latent = u_latent.expand(-1, input_seq.size(1), -1)  # (b, s, h)

        seq_embeddings = torch.cat([seq_embeddings, u_latent], dim=2).reshape(
            input_seq.size(0), -1, self.hidden_units
        )
        seq_embeddings = seq_embeddings + positional_embeddings

        # dropout
        if training:
            seq_embeddings = self.dropout_layer(seq_embeddings)

        # masking
        seq_embeddings = seq_embeddings * mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings  # (b, s, h1 + h2)

        seq_attention = self.encoder(seq_attention, training=training, mask=mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = pos * (pos != 0).long()  # masking
        neg = neg * (neg != 0).long()  # masking

        user_emb = u_latent.reshape(
            input_seq.size(0) * self.seq_max_len, self.user_embedding_dim
        )
        pos = pos.reshape(input_seq.size(0) * self.seq_max_len)
        neg = neg.reshape(input_seq.size(0) * self.seq_max_len)
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)

        # Add user embeddings
        pos_emb = torch.cat([pos_emb, user_emb], dim=1).reshape(-1, self.hidden_units)
        neg_emb = torch.cat([neg_emb, user_emb], dim=1).reshape(-1, self.hidden_units)

        seq_emb = seq_attention.reshape(
            input_seq.size(0) * self.seq_max_len, self.hidden_units
        )  # (b*s, d)

        pos_logits = (pos_emb * seq_emb).sum(dim=-1)
        neg_logits = (neg_emb * seq_emb).sum(dim=-1)

        pos_logits = pos_logits.unsqueeze(-1)  # (bs, 1)
        neg_logits = neg_logits.unsqueeze(-1)  # (bs, 1)

        # masking for loss calculation
        istarget = (pos != 0).float()

        return pos_logits, neg_logits, istarget

    def predict(self, inputs):
        """
        Model prediction for candidate (negative) items
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            user = inputs["user"]
            input_seq = inputs["input_seq"]
            candidate = inputs["candidate"]

            # Convert numpy arrays to tensors if necessary
            if not isinstance(user, torch.Tensor):
                user = torch.LongTensor(user).to(device)
            if not isinstance(input_seq, torch.Tensor):
                input_seq = torch.LongTensor(input_seq).to(device)
            if not isinstance(candidate, torch.Tensor):
                candidate = torch.LongTensor(candidate).to(device)

            mask = (input_seq != 0).float().unsqueeze(-1)
            seq_embeddings, positional_embeddings = self.embedding(input_seq)  # (1, s, h)

            u0_latent = self.user_embedding_layer(user)
            u0_latent = u0_latent * (self.user_embedding_dim ** 0.5)  # (1, 1, h)
            u0_latent = u0_latent.squeeze(0)  # (1, h)
            test_user_emb = u0_latent.expand(1 + self.num_neg_test, -1)  # (101, h)

            u_latent = self.user_embedding_layer(user)
            u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
            u_latent = u_latent.expand(-1, input_seq.size(1), -1)  # (b, s, h)

            seq_embeddings = torch.cat([seq_embeddings, u_latent], dim=2).reshape(
                input_seq.size(0), -1, self.hidden_units
            )
            seq_embeddings = seq_embeddings + positional_embeddings  # (b, s, h1 + h2)

            seq_embeddings = seq_embeddings * mask
            seq_attention = seq_embeddings
            seq_attention = self.encoder(seq_attention, training=False, mask=mask)
            seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)
            seq_emb = seq_attention.reshape(
                input_seq.size(0) * self.seq_max_len, self.hidden_units
            )  # (b*s1, h1+h2)

            candidate_emb = self.item_embedding_layer(candidate)  # (b, s2, h2)
            candidate_emb = candidate_emb.squeeze(0)  # (s2, h2)
            candidate_emb = torch.cat([candidate_emb, test_user_emb], dim=1).reshape(
                -1, self.hidden_units
            )  # (b*s2, h1+h2)

            candidate_emb = candidate_emb.transpose(0, 1)  # (h1+h2, b*s2)
            test_logits = torch.matmul(seq_emb, candidate_emb)  # (b*s1, b*s2)

            test_logits = test_logits.reshape(
                input_seq.size(0), self.seq_max_len, 1 + self.num_neg_test
            )  # (1, s, 101)
            test_logits = test_logits[:, -1, :]  # (1, 101)

        return test_logits

    def loss_function(self, pos_logits, neg_logits, istarget):
        """Losses are calculated separately for the positive and negative
        items based on the corresponding logits. A mask is included to
        take care of the zero items (added for padding).

        Args:
            pos_logits (torch.Tensor): Logits of the positive examples.
            neg_logits (torch.Tensor): Logits of the negative examples.
            istarget (torch.Tensor): Mask for nonzero targets.

        Returns:
            torch.Tensor: Loss.
        """
        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # ignore padding items (0)
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        # L2 regularization is handled by optimizer weight_decay
        return loss

    def create_combined_dataset(self, u, seq, pos, neg):
        """
        function to create model inputs from sampled batch data.
        This function is used only during training.
        Overrides parent to include users in the inputs.
        """
        from recommenders.models.sasrec.model import pad_sequences
        import numpy as np

        inputs = {}
        seq = pad_sequences(seq, padding="pre", truncating="pre", maxlen=self.seq_max_len)
        pos = pad_sequences(pos, padding="pre", truncating="pre", maxlen=self.seq_max_len)
        neg = pad_sequences(neg, padding="pre", truncating="pre", maxlen=self.seq_max_len)

        inputs["users"] = np.expand_dims(np.array(u), axis=-1)
        inputs["input_seq"] = seq
        inputs["positive"] = pos
        inputs["negative"] = neg

        target = np.concatenate(
            [
                np.repeat(1, seq.shape[0] * seq.shape[1]),
                np.repeat(0, seq.shape[0] * seq.shape[1]),
            ],
            axis=0,
        )
        target = np.expand_dims(target, axis=-1)
        return inputs, target

    def train_model(self, dataset, sampler, **kwargs):
        """
        High level function for model training as well as
        evaluation on the validation and test dataset.
        Overrides parent to include users in the input tensors.
        """
        from tqdm import tqdm
        from recommenders.utils.timer import Timer

        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("learning_rate", 0.001)
        val_epoch = kwargs.get("val_epoch", 5)

        num_steps = int(len(dataset.user_train) / batch_size)

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=self.l2_reg,
        )

        T = 0.0
        t0 = Timer()
        t0.start()

        for epoch in range(1, num_epochs + 1):
            step_loss = []
            self.train()  # Set training mode

            for _ in tqdm(
                range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
            ):
                u, seq, pos, neg = sampler.next_batch()

                inputs, target = self.create_combined_dataset(u, seq, pos, neg)

                # Convert to tensors and move to device
                users = torch.LongTensor(inputs["users"]).to(device)
                input_seq = torch.LongTensor(inputs["input_seq"]).to(device)
                positive = torch.LongTensor(inputs["positive"]).to(device)
                negative = torch.LongTensor(inputs["negative"]).to(device)

                inp = {
                    "users": users,
                    "input_seq": input_seq,
                    "positive": positive,
                    "negative": negative,
                }

                optimizer.zero_grad()
                pos_logits, neg_logits, loss_mask = self(inp, training=True)
                loss = self.loss_function(pos_logits, neg_logits, loss_mask)
                loss.backward()
                optimizer.step()

                step_loss.append(loss.item())

            if epoch % val_epoch == 0:
                t0.stop()
                t1 = t0.interval
                T += t1
                print("Evaluating...")
                t_test = self.evaluate(dataset)
                t_valid = self.evaluate_valid(dataset)
                print(
                    f"\nepoch: {epoch}, time: {T}, valid (NDCG@10: {t_valid[0]}, HR@10: {t_valid[1]})"
                )
                print(
                    f"epoch: {epoch}, time: {T},  test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})"
                )
                t0.start()

        t_test = self.evaluate(dataset)
        print(f"\nepoch: {epoch}, test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})")

        return t_test
