# Copyright © 2026 Apple Inc.

"""Automatic prefix caching shared by MLX language-model serving paths.

This module gives the existing radix-backed prompt cache a model-independent
APC interface.  Cache topology remains owned by ``model.make_cache()``: plain
KV, rotating/full hybrids (Laguna, North, and Gemma/Muse text backbones), and
checkpointed recurrent hybrids all use the same lookup and storage policy.

The implementation stores whole prompt-cache snapshots.  The radix indexes
token sequences; it is not a paged-KV allocator and does not claim block-level
copy-on-write sharing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, List, Optional

from .models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    LRUPromptCache,
    RotatingKVCache,
    achievable_trim,
    can_trim_prompt_cache,
)


@dataclass(frozen=True)
class APCKey:
    """Fail-closed identity for a reusable prompt-cache namespace.

    ``semantic_fingerprint`` is where multimodal callers bind image/audio or
    other non-token inputs.  It must change whenever equal token IDs could
    produce different model state.
    """

    model: Hashable
    revision: Optional[Hashable] = None
    adapter: Optional[Hashable] = None
    tokenizer_fingerprint: Optional[Hashable] = None
    cache_layout_fingerprint: Optional[Hashable] = None
    semantic_fingerprint: Optional[Hashable] = None


@dataclass(frozen=True)
class APCCapabilities:
    topology: str
    exact_prefix: bool
    arbitrary_branch: bool
    reason: Optional[str] = None


@dataclass
class APCLookup:
    cache: Optional[List[Any]]
    remaining_tokens: List[int]
    cached_tokens: int
    hit: bool
    hit_kind: Optional[str]
    miss_reason: Optional[str]


def _walk_cache_entries(prompt_cache: Iterable[Any]):
    for entry in prompt_cache:
        if isinstance(entry, CacheList):
            yield from _walk_cache_entries(entry.caches)
        elif isinstance(entry, (list, tuple)):
            yield from _walk_cache_entries(entry)
        else:
            yield entry


def inspect_apc_capabilities(prompt_cache: List[Any]) -> APCCapabilities:
    """Describe which lossless APC operations a concrete cache supports."""

    leaves = list(_walk_cache_entries(prompt_cache))
    if not leaves:
        return APCCapabilities("empty", False, False, "empty_cache")

    has_rotating = any(isinstance(c, RotatingKVCache) for c in leaves)
    has_full = any(isinstance(c, KVCache) for c in leaves)
    has_state = any(isinstance(c, ArraysCache) for c in leaves)
    known = all(
        isinstance(c, (KVCache, RotatingKVCache, ArraysCache))
        or hasattr(c, "state")
        for c in leaves
    )

    if has_state:
        topology = "checkpointed_hybrid"
    elif has_rotating and has_full:
        topology = "mixed_rotating_kv"
    elif has_rotating:
        topology = "rotating_kv"
    elif has_full:
        topology = "kv"
    else:
        topology = "custom"

    arbitrary_branch = can_trim_prompt_cache(prompt_cache)
    if not arbitrary_branch:
        # Checkpoint-aware hybrids can still branch at recorded positions.
        arbitrary_branch = achievable_trim(prompt_cache, 1) is not None

    return APCCapabilities(
        topology=topology,
        exact_prefix=known,
        arbitrary_branch=arbitrary_branch,
        reason=None if known else "unsupported_cache_entry",
    )


class AutomaticPrefixCache(LRUPromptCache):
    """Shared radix APC for standard and model-defined MLX cache topologies.

    The legacy ``fetch_nearest_cache`` / ``insert_cache`` methods remain
    available, so this is a drop-in replacement for ``LRUPromptCache``.  New
    callers should use ``lookup`` / ``store`` for explicit hit telemetry.
    """

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63):
        super().__init__(max_size=max_size, max_bytes=max_bytes)
        self._apc_stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "cached_tokens": 0,
            "stores": 0,
        }

    @staticmethod
    def key(
        model: Hashable,
        *,
        revision: Optional[Hashable] = None,
        adapter: Optional[Hashable] = None,
        tokenizer_fingerprint: Optional[Hashable] = None,
        cache_layout_fingerprint: Optional[Hashable] = None,
        semantic_fingerprint: Optional[Hashable] = None,
    ) -> APCKey:
        return APCKey(
            model=model,
            revision=revision,
            adapter=adapter,
            tokenizer_fingerprint=tokenizer_fingerprint,
            cache_layout_fingerprint=cache_layout_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
        )

    def lookup(self, key: Hashable, tokens: Iterable[int]) -> APCLookup:
        tokens = [int(token) for token in tokens]
        trie_result = self._trie.search(key, tokens)
        cache, remaining = super().fetch_nearest_cache(key, tokens)
        cached_tokens = len(tokens) - len(remaining) if cache is not None else 0
        hit = cache is not None and cached_tokens > 0

        self._apc_stats["lookups"] += 1
        self._apc_stats["hits" if hit else "misses"] += 1
        self._apc_stats["cached_tokens"] += cached_tokens

        if not hit:
            kind = None
            short_length = (
                len(trie_result.shorter) if trie_result.shorter is not None else 0
            )
            has_unusable_branch = trie_result.exact is not None or (
                trie_result.longer is not None
                and trie_result.common_prefix > short_length
            )
            reason = (
                "untrimmable_branch"
                if has_unusable_branch
                else "no_compatible_prefix"
            )
        elif trie_result.exact is not None:
            kind = "exact"
            reason = None
        else:
            kind = "prefix"
            reason = None
        return APCLookup(cache, remaining, cached_tokens, hit, kind, reason)

    def store(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ) -> APCCapabilities:
        capabilities = inspect_apc_capabilities(prompt_cache)
        if not capabilities.exact_prefix:
            return capabilities
        super().insert_cache(
            key, [int(token) for token in tokens], prompt_cache, cache_type=cache_type
        )
        self._apc_stats["stores"] += 1
        return capabilities

    def fetch_nearest_cache(self, model: Hashable, tokens: List[int]):
        result = self.lookup(model, tokens)
        return result.cache, result.remaining_tokens

    def insert_cache(
        self,
        model: Hashable,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        return self.store(model, tokens, prompt_cache, cache_type=cache_type)

    @property
    def apc_stats(self):
        return dict(self._apc_stats)


# Short public spelling for server integrations.
APC = AutomaticPrefixCache


__all__ = [
    "APC",
    "APCCapabilities",
    "APCKey",
    "APCLookup",
    "AutomaticPrefixCache",
    "inspect_apc_capabilities",
]
