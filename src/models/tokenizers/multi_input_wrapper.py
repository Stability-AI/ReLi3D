from dataclasses import dataclass, field

import torch
import torch.nn as nn

import src
from src.constants import FieldName, Names, OutputsType
from src.utils.typing import List

from .abstract_tokenizer import AbstractTokenizer


class MultiInputTokenizerWrapper(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)
        input_keys: List[FieldName] = field(default_factory=list)
        fixed_conditioning_keys: List[FieldName] = field(default_factory=list)
        tokenizer_input_key: FieldName = Names.IMAGE.cond
        embedding_dimension: int = 0

    cfg: Config

    def configure(self):
        super().configure()

        self.tokenizer = src.initialize_instance(
            self.cfg.tokenizer_cls, self.cfg.tokenizer
        )

        self.num_inputs = len(self.cfg.input_keys)

        if self.cfg.embedding_dimension > 0:
            self.index_embedding = torch.nn.Parameter(
                torch.randn(len(self.cfg.input_keys), self.cfg.embedding_dimension)
            )

    def forward(self, outputs: OutputsType) -> OutputsType:
        joined_input = torch.cat([outputs[key] for key in self.cfg.input_keys], dim=0)
        joined_tokens = self.tokenizer(
            {
                k: v.repeat(self.num_inputs, *[1 for _ in v.shape[1:]])
                if isinstance(v, torch.Tensor)
                else v
                for k, v in outputs.items()
            }  # Double batch dim
            | {self.cfg.tokenizer_input_key: joined_input}
        )[self.cfg.tokenize_key]
        individual_tokens = joined_tokens.chunk(len(self.cfg.input_keys), dim=0)
        if self.cfg.embedding_dimension > 0:
            individual_tokens = [
                torch.cat(
                    [
                        token,
                        embedding.view(*[1 for _ in token.shape[:-1]], -1).expand(
                            *token.shape[:-1], -1
                        ),
                    ],
                    dim=-1,
                )
                for token, embedding in zip(individual_tokens, self.index_embedding)
            ]
        token_dict = {
            key.add_suffix("tokenized"): token.contiguous()
            for key, token in zip(self.cfg.input_keys, individual_tokens)
        }

        return token_dict


class RandomMaskTokenizerWrapper(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)
        image_key: FieldName = Names.IMAGE.cond
        mask_key: FieldName = Names.OPACITY.cond
        dropout_prob: float = 0.2
        use_mask: bool = True

    cfg: Config

    def configure(self):
        super().configure()
        self.tokenizer = src.initialize_instance(
            self.cfg.tokenizer_cls, self.cfg.tokenizer
        )

    def forward(self, outputs: OutputsType, use_mask: bool = True) -> OutputsType:
        # If we don't dropout the mask will be full white. If we dropout we apply the mask to the image.
        # We also concat the mask to the image and feed it to the tokenizer.
        image = outputs[self.cfg.image_key]
        mask = outputs[self.cfg.mask_key]
        full_mask = torch.ones_like(mask)
        batch_size = image.shape[0]

        if self.training:
            # Generate random mask per batch element
            should_mask = (
                torch.rand(batch_size, device=image.device) < self.cfg.dropout_prob
            )
            # Expand should_mask to match image dimensions
            should_mask = should_mask.view(batch_size, *[1 for _ in mask.shape[1:]])
            # Apply mask selectively to batch elements
            mask = torch.where(
                should_mask,
                mask,
                full_mask,
            )
        else:
            if not use_mask or not self.cfg.use_mask:
                mask = full_mask

        image = (image * mask).detach()

        # Pass through tokenizer
        tokenizer_outputs = self.tokenizer(
            outputs | {self.cfg.image_key: image, self.cfg.mask_key: mask}
        )

        return tokenizer_outputs

    def consumed_keys(self):
        return (
            super()
            .consumed_keys()
            .union(
                self.tokenizer.consumed_keys(), {self.cfg.image_key, self.cfg.mask_key}
            )
        )

    def produced_keys(self):
        return super().produced_keys().union(self.tokenizer.produced_keys())


class MixtureOfExpertsMultiViewTokenizerWrapper(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)
        additional_keys: List[FieldName] = field(default_factory=list)
        additional_dims: List[int] = field(default_factory=list)
        hidden_dim: int = 1024
        tokenizer_output_dim: int = -1

    cfg: Config

    def configure(self):
        super().configure()
        self.tokenizer = src.initialize_instance(
            self.cfg.tokenizer_cls, self.cfg.tokenizer
        )
        self.tokenizer_output_dim = self.cfg.tokenizer_output_dim
        if self.tokenizer_output_dim == -1:
            raise ValueError("tokenizer_output_dim must be set")
        self.tokenizer_output_key = self.tokenizer.cfg.tokenize_key

        self.total_input_dim = sum(self.cfg.additional_dims) + self.tokenizer_output_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.total_input_dim, self.cfg.hidden_dim),  # Combine both
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, 1),  # Output scalar score per image
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        batch_size = outputs[Names.BATCH_SIZE]
        tokenizer_output = self.tokenizer(outputs)
        tokens = tokenizer_output[self.tokenizer_output_key]
        token_meaned = tokens.mean(-2)

        additional_inputs = torch.cat(
            [outputs[key] for key in self.cfg.additional_keys], dim=0
        )

        gate_output = self.gate_mlp(
            torch.cat([token_meaned, additional_inputs], dim=-1)
        ).view(batch_size, -1)
        gate_output = torch.softmax(gate_output, dim=-1)

        final_tokens = (
            tokens
            * gate_output.view(
                batch_size, gate_output.shape[-1], *[1 for _ in tokens.shape[2:]]
            )
        ).sum(1, keepdim=True)

        return {self.tokenizer_output_key: final_tokens}

    def consumed_keys(self):
        return super().consumed_keys().union(self.tokenizer.consumed_keys())

    def produced_keys(self):
        return super().produced_keys().union(self.tokenizer_output_key)


class SoftClusterAttentionMultiViewTokenizerWrapper(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)
        additional_keys: List[FieldName] = field(default_factory=list)
        additional_dims: List[int] = field(default_factory=list)
        hidden_dim: int = 1024
        num_heads: int = 8
        num_clusters: int = 256
        tokenizer_output_dim: int = -1
        final_tokens: int = 128  # example: fewer than num_clusters if you want
        pose_embed_dim: int = 1024  # how large to project camera poses
        aggregator_num_heads: int = 4  # for aggregator attention

    def configure(self):
        super().configure()

        # ---------------------
        # Instantiate tokenizer
        # ---------------------
        self.tokenizer = src.initialize_instance(
            self.cfg.tokenizer_cls, self.cfg.tokenizer
        )
        self.tokenizer_output_key = self.tokenizer.cfg.tokenize_key

        if self.cfg.tokenizer_output_dim == -1:
            raise ValueError("tokenizer_output_dim must be set")
        self.tokenizer_output_dim = self.cfg.tokenizer_output_dim

        # ---------------------------------------
        # 1) Small MLP to embed camera poses, etc.
        # ---------------------------------------
        self.total_additional_dim = sum(self.cfg.additional_dims)

        # If you store per-image data of shape (B, num_images, <dim>),
        # we can project it into the same dimension as the tokens:
        self.pose_embedder = nn.Sequential(
            nn.Linear(self.total_additional_dim, self.cfg.pose_embed_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.pose_embed_dim, self.tokenizer_output_dim),
        )

        # -----------------------------------
        # 2) Cross-image token attention
        # -----------------------------------
        self.cross_image_attention = nn.MultiheadAttention(
            self.tokenizer_output_dim, self.cfg.num_heads, batch_first=True
        )

        # -----------------------------------------
        # 3) An "image‐level aggregator" attention
        # -----------------------------------------
        # We'll do a small self‐attention that pools tokens within *one* image
        # down to a single descriptor. This is better than plain "mean" because
        # it can learn to focus on more salient tokens.
        self.per_image_aggregator = nn.MultiheadAttention(
            self.tokenizer_output_dim, self.cfg.aggregator_num_heads, batch_first=True
        )

        # -------------------------------------------
        # 4) Another aggregator across the "num_images"
        # -------------------------------------------
        # This is to fuse the per‐image descriptors into a single set of cluster logits.
        # We'll keep it simple: produce one "global aggregator token" per batch
        # that attends to the image descriptors.
        self.global_aggregator_token = nn.Parameter(
            torch.randn(1, 1, self.tokenizer_output_dim)
        )
        self.image_fuser_attention = nn.MultiheadAttention(
            self.tokenizer_output_dim, self.cfg.aggregator_num_heads, batch_first=True
        )

        # Maps the final fused descriptor to cluster logits:
        self.fused_to_clusters = nn.Sequential(
            nn.Linear(self.tokenizer_output_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_clusters),
        )

        # ----------------------------------
        # 5) Learned cluster tokens
        # ----------------------------------
        self.cluster_tokens = nn.Parameter(
            torch.randn(1, self.cfg.num_clusters, self.tokenizer_output_dim)
        )

        # And the final attention that picks out the final tokens
        self.final_attention = nn.MultiheadAttention(
            self.tokenizer_output_dim, self.cfg.num_heads, batch_first=True
        )

    def forward(self, outputs):
        """
        Expecting:
          - outputs: dict with
              outputs[Names.BATCH_SIZE] = B
              outputs[<image data keys>]
              for each key in self.cfg.additional_keys, shape is (B, num_images, dim)
        """

        # batch_size = outputs[Names.BATCH_SIZE]

        # -------------------------------------------------
        # Tokenize: (B, num_images, num_tokens, embed_dim)
        # -------------------------------------------------
        tokenizer_output = self.tokenizer(outputs)
        tokens = tokenizer_output[self.tokenizer_output_key]
        # tokens: [batch_size, num_images, num_tokens, tokenizer_output_dim]

        B, num_images, num_tokens, D = tokens.shape

        # --------------------------------
        # (Optional) Add camera pose info
        # --------------------------------
        # We embed each image's additional data into an embedding of size D
        # and simply add it to each token in that image:
        # shape: (B, num_images, total_additional_dim)
        if self.cfg.additional_keys:
            additional_inputs = torch.cat(
                [outputs[key] for key in self.cfg.additional_keys], dim=-1
            )  # (B, num_images, sum_of_additional_dims)
            pose_embeds = self.pose_embedder(additional_inputs)  # (B, num_images, D)

            # Reshape so we can broadcast-add to each token
            pose_embeds_expanded = pose_embeds.unsqueeze(2)  # (B, num_images, 1, D)
            pose_embeds_expanded = pose_embeds_expanded.expand(-1, -1, num_tokens, -1)
            tokens = tokens + pose_embeds_expanded  # inject pose info

        # ----------------------------------------------------------------
        # Step 1: Cross-image attention across *all tokens from all images*
        # Flatten across images: shape => (B, num_images * num_tokens, D)
        # ----------------------------------------------------------------
        flattened_tokens = tokens.view(B, num_images * num_tokens, D)
        cross_image_tokens, _ = self.cross_image_attention(
            flattened_tokens, flattened_tokens, flattened_tokens
        )
        # Reshape back to (B, num_images, num_tokens, D)
        cross_image_tokens = cross_image_tokens.view(B, num_images, num_tokens, D)

        # -----------------------------------------------------------------
        # Step 2: "Per-image aggregator" => produce one descriptor per image
        # -----------------------------------------------------------------
        # We'll do a per‐image self‐attention that yields 1 token per image:
        # 1) We'll create a small "aggregator token" for each image,
        #    or we can simply ask for the "cls" in the multi-head attention approach
        #    by prepending a learned token. For simplicity, let's do the "prepend token."
        aggregator_token = nn.Parameter(
            torch.zeros(1, 1, D).to(tokens.device)
        )  # shape (1,1,D)

        # Expand aggregator_token for each image in the batch
        # We'll have shape: (B*num_images, 1, D)
        aggregator_token_expanded = aggregator_token.expand(
            B * num_images, 1, D
        ).clone()

        # Flatten cross_image_tokens => shape (B*num_images, num_tokens, D)
        cross_image_tokens_flat = cross_image_tokens.reshape(
            B * num_images, num_tokens, D
        )

        # Concatenate aggregator_token at index 0 of the sequence
        aggregator_input = torch.cat(
            [aggregator_token_expanded, cross_image_tokens_flat], dim=1
        )
        # aggregator_input => (B*num_images, 1 + num_tokens, D)

        # Self-attention:
        # queries, keys, values are aggregator_input
        # We'll only slice out the aggregator_token part after attention
        aggregator_output, _ = self.per_image_aggregator(
            aggregator_input, aggregator_input, aggregator_input
        )
        # aggregator_output shape => (B*num_images, 1 + num_tokens, D)
        # aggregator_token is aggregator_output[:, 0, :] => the first token

        aggregator_token_final = aggregator_output[:, 0, :].view(B, num_images, D)
        # shape => (B, num_images, D), one descriptor per image

        # -----------------------------------------------------------------
        # Step 3: Aggregator across images => produce single "fused descriptor"
        # -----------------------------------------------------------------
        # We'll do an attention from a single "global aggregator token" to
        # the per-image descriptors. That yields a single descriptor for the entire batch.
        # For a more flexible approach, you could replicate a "global aggregator" for each
        # batch element or do something else. We'll keep it simple:
        global_token = self.global_aggregator_token.expand(B, 1, D)  # (B, 1, D)

        # aggregator_token_final => (B, num_images, D)
        # We'll treat it as (B, num_images, D) and let the global_token attend.
        fused_output, _ = self.image_fuser_attention(
            global_token,  # queries => shape (B, 1, D)
            aggregator_token_final,  # keys => shape (B, num_images, D)
            aggregator_token_final,  # values => shape (B, num_images, D)
        )
        # fused_output => (B, 1, D)

        fused_descriptor = fused_output.squeeze(1)  # => (B, D)

        # Map to cluster logits
        cluster_logits = self.fused_to_clusters(fused_descriptor)  # (B, num_clusters)
        cluster_weights = torch.softmax(cluster_logits, dim=-1)  # (B, num_clusters)

        # -------------------------------------------------
        # Step 4: Create "dynamic queries" from cluster_tokens
        # -------------------------------------------------
        # cluster_tokens => shape (1, num_clusters, D)
        # Expand to (B, num_clusters, D), then do weighted sum:
        cluster_tokens = self.cluster_tokens.expand(B, -1, -1)
        # Weighted sum => dynamic queries
        dynamic_queries = torch.einsum("bn,bnd->bnd", cluster_weights, cluster_tokens)
        # shape => (B, num_clusters, D)

        # Flatten cross_image_tokens again => (B, num_images*num_tokens, D)
        all_tokens = cross_image_tokens.view(B, -1, D)

        # Final attention from the dynamic queries to all tokens:
        # If you want only cfg.final_tokens output, slice the queries
        # to that many tokens.
        query_slice = dynamic_queries[:, : self.cfg.final_tokens, :]
        selected_tokens, _ = self.final_attention(
            query_slice,  # queries
            all_tokens,  # keys
            all_tokens,  # values
        )
        # selected_tokens => (B, final_tokens, D)

        # Return the new tokens
        return {self.tokenizer_output_key: selected_tokens}

    def consumed_keys(self):
        return (
            super()
            .consumed_keys()
            .union(self.tokenizer.consumed_keys(), self.cfg.additional_keys)
        )

    def produced_keys(self):
        return super().produced_keys().union(self.tokenizer_output_key)
