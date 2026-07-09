"""Hybrid retrieval proposer for draft-free prompt-lookup decoding.

This opt-in module adds a SAM-Decoding style **datastore** proposer: a static suffix automaton
built once over a persistent corpus (previous prompts/completions, a code index,
a canned prelude, ...) that proposes continuations even when the in-context
matcher has never seen the pattern this session.

Three classes, all speaking the shipped proposer interface
(``observe(token)`` / ``propose(seq, max_span, prompt_len) -> list[int]``, see
``mlx_lm/generate.py::prompt_lookup_generate_step``):

- ``DatastoreProposer`` — static suffix automaton over a fixed datastore. Given
  the running sequence, it finds the longest suffix of that sequence which also
  occurs in the datastore and returns the datastore's continuation. Stateless
  w.r.t. the generation (``observe`` is a no-op): it matches on the passed-in
  ``seq`` tail every call, so its coordinates never desync from the caller.

- ``HybridProposer`` — tries an in-context primary proposer first (the shipped
  ``SuffixAutomatonProposer`` / ``NgramProposer``, which retrieve from *this*
  session) and falls back to the datastore only when the primary has no match.
  This is the SAM-Decoding fallback ordering: dynamic in-context first, static
  corpus second.

Optional ``adaptive_span`` (LogitSpec / AdaPLD flavour): scale the proposed
draft length by the match length, so short/uncertain matches propose fewer
tokens (cheaper mis-speculation) and long confident matches propose the full
budget. Construction deduplicates exact documents by default, exposes corpus /
automaton footprint statistics, and avoids retaining a second copy of the token
store beyond the suffix automaton's own sequence.

Reuses the shipped ``SuffixAutomaton`` verbatim for construction; the datastore
query walk is added here because the shipped automaton only answers "longest
repeated suffix of my own last-appended token" — a datastore must instead match
an *external* query tail against a *frozen* corpus.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from itertools import chain
from numbers import Integral
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

try:  # normal package import
    from .prompt_lookup import NgramProposer, SuffixAutomaton, SuffixAutomatonProposer
except ImportError:  # standalone (tests load the module by file path)
    from prompt_lookup import (  # type: ignore
        NgramProposer,
        SuffixAutomaton,
        SuffixAutomatonProposer,
    )

# Sentinel token inserted between datastore documents so a match/continuation
# never bleeds across a document boundary. Real token ids are non-negative, so a
# negative id is safe and impossible to collide with.
_DOC_SEP = -1

__all__ = [
    "DatastoreProposer",
    "DatastoreStats",
    "HybridProposer",
    "HybridProposerStats",
    "make_hybrid_proposer",
]


@dataclass(frozen=True)
class DatastoreStats:
    """Immutable construction statistics for a :class:`DatastoreProposer`."""

    input_documents: int
    documents: int
    duplicate_documents: int
    tokens: int
    automaton_states: int
    automaton_transitions: int


@dataclass
class HybridProposerStats:
    """Source-level retrieval telemetry for workload A/Bs."""

    calls: int = 0
    primary_hits: int = 0
    datastore_hits: int = 0
    misses: int = 0
    primary_proposed: int = 0
    datastore_proposed: int = 0
    datastore_suppressed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


class DatastoreProposer:
    """Static suffix-automaton retrieval over a persistent datastore.

    Parameters
    ----------
    datastore:
        Either an iterable of token-id sequences (each a document) or a single
        flat token-id sequence. Documents are separated internally so matches
        and continuations never cross a boundary.
    min_match:
        Minimum matched-suffix length (in tokens) before a proposal is made.
        Longer values trade recall for precision (fewer wasted verifications).
    window:
        Only the last ``window`` tokens of the running sequence are walked
        against the automaton each call. Bounds per-step cost to O(window);
        must be >= the longest match you care to find.
    adaptive_span:
        When True, the number of proposed tokens is capped at the match length
        times ``span_scale`` (AdaPLD-style). The result is still clamped by the
        caller's ``max_span`` and is never smaller than ``min_span``.
    deduplicate:
        Skip exact duplicate documents. This reduces footprint without changing
        the set of distinct datastore substrings or their first continuation.
    max_tokens:
        Optional fail-closed construction budget, excluding separators. The
        builder rejects a corpus that exceeds it rather than silently truncating
        a document and introducing a synthetic continuation.
    sep:
        Document-separator sentinel (default ``-1``).
    """

    __slots__ = (
        "min_match",
        "window",
        "adaptive_span",
        "span_scale",
        "min_span",
        "sep",
        "sam",
        "stats",
    )

    def __init__(
        self,
        datastore: Iterable[Sequence[int]] | Sequence[int],
        *,
        min_match: int = 3,
        window: int = 64,
        adaptive_span: bool = False,
        span_scale: float = 1.0,
        min_span: int = 1,
        deduplicate: bool = True,
        max_tokens: int | None = None,
        sep: int = _DOC_SEP,
    ) -> None:
        if min_match < 1:
            raise ValueError("min_match must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not math.isfinite(span_scale) or span_scale <= 0:
            raise ValueError("span_scale must be finite and > 0")
        if min_span < 1:
            raise ValueError("min_span must be >= 1")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if sep >= 0:
            raise ValueError("sep must be a negative token id")
        self.min_match = int(min_match)
        self.window = int(window)
        self.adaptive_span = bool(adaptive_span)
        self.span_scale = float(span_scale)
        self.min_span = int(min_span)
        self.sep = int(sep)
        self.sam = SuffixAutomaton()
        seen = set()
        input_documents = 0
        documents = 0
        duplicate_documents = 0
        token_count = 0
        for doc in _iter_documents(datastore):
            input_documents += 1
            tokens = tuple(_validated_token(t, self.sep) for t in doc)
            if not tokens:
                continue
            if deduplicate and tokens in seen:
                duplicate_documents += 1
                continue
            if max_tokens is not None and token_count + len(tokens) > max_tokens:
                raise ValueError(
                    f"datastore token budget exceeded: "
                    f"{token_count + len(tokens)} > {max_tokens}"
                )
            if deduplicate:
                seen.add(tokens)
            for t in tokens:
                self.sam.extend(t)
            # Separate documents so a suffix at the end of doc A cannot propose
            # a continuation that starts in doc B.
            self.sam.extend(self.sep)
            documents += 1
            token_count += len(tokens)
        self.stats = DatastoreStats(
            input_documents=input_documents,
            documents=documents,
            duplicate_documents=duplicate_documents,
            tokens=token_count,
            automaton_states=len(self.sam._len),
            automaton_transitions=sum(len(trans) for trans in self.sam._next),
        )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        tokenizer: Any,
        *,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> "DatastoreProposer":
        """Tokenize text documents lazily and build a datastore.

        ``tokenizer`` may expose ``encode`` or be a callable returning token ids.
        Special tokens default off so document boundaries remain owned solely by
        the datastore separator.
        """

        def tokenized() -> Iterator[Sequence[int]]:
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError("datastore texts must be strings")
                if hasattr(tokenizer, "encode"):
                    try:
                        yield tokenizer.encode(
                            text, add_special_tokens=add_special_tokens
                        )
                    except TypeError:
                        yield tokenizer.encode(text)
                elif callable(tokenizer):
                    yield tokenizer(text)
                else:
                    raise TypeError("tokenizer must be callable or expose encode()")

        return cls(tokenized(), **kwargs)

    def footprint_bytes(self) -> int:
        """Best-effort deep Python heap footprint of the frozen automaton.

        This intentionally excludes interpreter/module overhead and is suitable
        for comparing datastore configurations in one process, not as RSS.
        """

        return _deep_size(self.sam)

    # -- proposer interface -------------------------------------------------
    def observe(self, token: int) -> None:  # datastore is frozen; stateless
        pass

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        if max_span <= 0 or not self.sam.seq:
            return []
        tail = seq[-self.window :]
        state, match_len = self._match(tail)
        if match_len < self.min_match:
            return []
        # First occurrence of this state's substrings ends at ``first_end``;
        # the continuation begins right after it in the datastore.
        end = self.sam._first_end[state]
        start = end + 1
        span = max_span
        if self.adaptive_span:
            scaled = max(self.min_span, math.ceil(match_len * self.span_scale))
            span = min(max_span, scaled)
        return self._continuation(start, span)

    # -- internals ----------------------------------------------------------
    def _match(self, query_tail: Sequence[int]) -> Tuple[int, int]:
        """Walk ``query_tail`` through the automaton. Return (state, match_len):
        the longest suffix of ``query_tail`` that is a substring of the datastore
        and the automaton state recognising it. (0, 0) when nothing matches."""
        nxt = self.sam._next
        link = self.sam._link
        lens = self.sam._len
        v, length = 0, 0
        for c in query_tail:
            c = int(c)
            # Cannot extend the current match: shorten via suffix links until a
            # state can consume ``c`` (or we fall back to the root).
            while v != 0 and c not in nxt[v]:
                v = link[v]
                length = lens[v]
            trans = nxt[v]
            if c in trans:
                v = trans[c]
                length += 1
            else:
                v, length = 0, 0
        return v, length

    def _continuation(self, start: int, span: int) -> List[int]:
        cont: List[int] = []
        store = self.sam.seq
        n = len(store)
        i = start
        while i < n and len(cont) < span:
            t = store[i]
            if t == self.sep:  # stop at the document boundary
                break
            cont.append(t)
            i += 1
        return cont


class HybridProposer:
    """In-context proposer first, datastore fallback second (SAM-Decoding order).

    ``primary`` is any shipped in-context proposer (``SuffixAutomatonProposer``
    or ``NgramProposer``) that retrieves from the *current* session; ``datastore``
    is a :class:`DatastoreProposer` over a persistent corpus. ``observe`` is
    forwarded to both so the primary's stateful automaton stays seeded exactly as
    the shipped code expects. ``datastore_warmup_tokens`` reserves an initial
    primary-only interval, while ``datastore_cooldown`` can suppress fallback for
    proposal cycles immediately after an in-context hit. Both guards default off
    because measured guard sweeps reduced wins on replay-friendly workloads.
    """

    __slots__ = (
        "primary",
        "datastore",
        "stats",
        "datastore_cooldown",
        "datastore_warmup_tokens",
        "_cooldown_remaining",
    )

    def __init__(
        self,
        primary,
        datastore: DatastoreProposer,
        stats: HybridProposerStats | None = None,
        datastore_cooldown: int = 0,
        datastore_warmup_tokens: int = 0,
    ) -> None:
        if not hasattr(primary, "propose") or not hasattr(primary, "observe"):
            raise TypeError("primary must expose observe()/propose()")
        if datastore_cooldown < 0:
            raise ValueError("datastore_cooldown must be >= 0")
        if datastore_warmup_tokens < 0:
            raise ValueError("datastore_warmup_tokens must be >= 0")
        self.primary = primary
        self.datastore = datastore
        self.stats = stats if stats is not None else HybridProposerStats()
        self.datastore_cooldown = int(datastore_cooldown)
        self.datastore_warmup_tokens = int(datastore_warmup_tokens)
        self._cooldown_remaining = 0

    def observe(self, token: int) -> None:
        self.primary.observe(token)
        self.datastore.observe(token)

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        self.stats.calls += 1
        prop = self.primary.propose(seq, max_span, prompt_len)
        if prop:
            self.stats.primary_hits += 1
            self.stats.primary_proposed += len(prop)
            self._cooldown_remaining = self.datastore_cooldown
            return prop
        generated = max(len(seq) - prompt_len, 0)
        if generated < self.datastore_warmup_tokens:
            self.stats.datastore_suppressed += 1
            self.stats.misses += 1
            return []
        if self._cooldown_remaining:
            self._cooldown_remaining -= 1
            self.stats.datastore_suppressed += 1
            self.stats.misses += 1
            return []
        prop = self.datastore.propose(seq, max_span, prompt_len)
        if prop:
            self.stats.datastore_hits += 1
            self.stats.datastore_proposed += len(prop)
            return prop
        self.stats.misses += 1
        return []


def make_hybrid_proposer(
    datastore: Iterable[Sequence[int]] | Sequence[int],
    *,
    primary: str = "suffix_automaton",
    min_match: int = 3,
    window: int = 64,
    adaptive_span: bool = False,
    span_scale: float = 1.0,
    min_span: int = 1,
    deduplicate: bool = True,
    max_tokens: int | None = None,
    datastore_cooldown: int = 0,
    datastore_warmup_tokens: int = 0,
    primary_kwargs: Dict[str, Any] | None = None,
) -> HybridProposer:
    """Convenience builder. ``primary`` is ``"suffix_automaton"`` or ``"ngram"``.

    The returned proposer is *empty* of session state (like ``make_proposer``):
    the caller's ``observe`` loop is the single seeding authority, so do not feed
    the prompt here."""
    primary_kwargs = dict(primary_kwargs or {})
    if primary in (None, "suffix_automaton"):
        primary_proposer = SuffixAutomatonProposer(**primary_kwargs)
    elif primary == "ngram":
        primary_proposer = NgramProposer(**primary_kwargs)
    else:
        raise ValueError(f"unknown primary backend {primary!r}")
    ds = DatastoreProposer(
        datastore,
        min_match=min_match,
        window=window,
        adaptive_span=adaptive_span,
        span_scale=span_scale,
        min_span=min_span,
        deduplicate=deduplicate,
        max_tokens=max_tokens,
    )
    return HybridProposer(
        primary_proposer,
        ds,
        datastore_cooldown=datastore_cooldown,
        datastore_warmup_tokens=datastore_warmup_tokens,
    )


def _iter_documents(
    datastore: Iterable[Sequence[int]] | Sequence[int],
) -> Iterator[Iterable[int]]:
    """Yield documents without materializing an outer document iterator."""

    iterator = iter(datastore)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if _is_token(first):
        yield chain((first,), iterator)
        return
    if isinstance(first, (str, bytes)) or not hasattr(first, "__iter__"):
        raise TypeError("datastore must contain token ids or token-id documents")
    yield first
    for doc in iterator:
        if isinstance(doc, (str, bytes)) or not hasattr(doc, "__iter__"):
            raise TypeError("datastore documents must be iterables of token ids")
        yield doc


def _is_token(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validated_token(value: Any, sep: int) -> int:
    if not _is_token(value):
        raise TypeError(f"token ids must be integers, got {type(value).__name__}")
    token = int(value)
    if token < 0:
        if token == sep:
            raise ValueError("datastore document contains the separator token")
        raise ValueError("datastore token ids must be non-negative")
    return token


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Recursively count container/object storage once by identity."""

    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    elif hasattr(value, "__slots__"):
        for slot in value.__slots__:
            if hasattr(value, slot):
                size += _deep_size(getattr(value, slot), seen)
    elif hasattr(value, "__dict__"):
        size += _deep_size(vars(value), seen)
    return size
