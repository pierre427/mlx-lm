# Copyright © 2025 Apple Inc.

import os

import mlx.core as mx


class PipelineMixin:
    def __init__(self):
        super().__init__()
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = None

    @property
    def pipeline_layers(self):
        return self.layers[self.start_idx : self.end_idx]

    def pipeline(self, group):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        # MLX_PIPELINE_LAYERS="35,12" assigns explicit per-rank layer counts
        # (rank order), for heterogeneous machines. Must sum to len(layers).
        counts_env = os.environ.get("MLX_PIPELINE_LAYERS")
        if counts_env:
            counts = [int(c) for c in counts_env.split(",")]
            if len(counts) != self.pipeline_size or sum(counts) != len(self.layers):
                raise ValueError(
                    f"MLX_PIPELINE_LAYERS={counts_env} must list "
                    f"{self.pipeline_size} counts summing to {len(self.layers)}"
                )
            # rank r owns counts[r] layers; rank 0 takes the last block
            self.start_idx = sum(counts[self.pipeline_rank + 1 :])
            self.end_idx = self.start_idx + counts[self.pipeline_rank]
        else:
            layers_per_rank = len(self.layers) // self.pipeline_size
            extra = len(self.layers) - layers_per_rank * self.pipeline_size
            if self.pipeline_rank < extra:
                layers_per_rank += 1
            self.start_idx = (
                self.pipeline_size - self.pipeline_rank - 1
            ) * layers_per_rank
            self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        # Keep the layer numbers the same for model loading
        self.layers[: self.start_idx] = [None] * self.start_idx
