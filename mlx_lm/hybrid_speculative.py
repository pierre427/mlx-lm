# Copyright © 2023-2024 Apple Inc.

"""Hybrid speculative decoding: retrieval-first + confidence-gated draft chain.

Greedy / temperature-0, same-tokenizer only. Per cycle the proposer picks the
cheapest credible source of speculation:

  1. RETRIEVAL FIRST — a suffix automaton over (prompt + generated so far)
     finds the longest suffix of the committed sequence that occurred earlier;
     if the match is long enough (``min_match``) the continuation after the
     earlier occurrence is proposed verbatim, up to ``max_span`` tokens.
     Zero draft-model cost; the design center is copy-heavy output (quoting,
     diffs, structured repetition) where spans verify at ~5 tokens per
     weight-stream.
  2. ELSE a CONFIDENCE-GATED DRAFT CHAIN — the draft model runs
     autoregressively up to ``num_draft_tokens``, but drafting stops at the
     first token whose draft probability falls below ``tau``. Only the
     confident prefix is submitted for verification.
  3. If neither source proposes anything, a plain single-token target step is
     taken: no draft cost, no verify overhead — exactly baseline cost for
     that token. This is what lets the hybrid not lose on open-ended prose.

Verification is a single target forward over
``[pending committed tokens, proposal...]`` with standard greedy
longest-prefix acceptance plus the target's bonus/correction token —
identical semantics (and cache trim bookkeeping) to
``speculative_generate_step``. The draft model is optional: with
``draft_model=None`` this is pure prompt-lookup decoding (PLD) backed by a
suffix automaton.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .generate import (
    GenerationResponse,
    generate_step,
    generation_stream,
    wired_limit,
)
from .models.cache import (
    KVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper
from .prompt_lookup import HybridStats as _PromptLookupStatsBase
from .prompt_lookup import plan_proposal_around_verify_cliff
from .speculation_router import RoutedSpeculationPolicy

_GREEDY = make_sampler(temp=0.0)

# MTP rate gate (see _mtp_draft_verify_loop): one-shot measured keep-or-drop
# decision for the self-spec loop. Warmup cycles gather the spec-rate sample;
# the probe decodes that many tokens plainly (still delivered output); spec
# must beat plain by the margin to stay on.
_RATE_GATE_WARMUP_CYCLES = 8
_RATE_GATE_PROBE_TOKENS = 12
_RATE_GATE_MARGIN = 0.03


def _start_speculation_or_cleanup(caches, required_caches, error_message):
    """Enable rollback atomically and fail without leaking cache state."""
    try:
        for c in caches:
            c.start_speculation()
        if not can_trim_prompt_cache(required_caches):
            raise ValueError(error_message)
    except Exception:
        # Validation happens after start because recurrent caches only become
        # trimmable while recording rollback.  A failed capability gate or a
        # later cache's start failure must still unwind every earlier cache.
        for c in caches:
            try:
                c.stop_speculation()
            except Exception:
                pass
        raise


def _stop_all_speculation(caches):
    """Run ``stop_speculation`` on every cache even when one raises.

    A failing cleanup hook must not leave later caches speculating (holding
    rollback stashes and reporting trimmable state). Every hook runs; the
    first error is re-raised only after all caches had their turn.
    """
    first_error = None
    for c in caches:
        try:
            c.stop_speculation()
        except BaseException as e:  # noqa: BLE001 — cleanup must reach every cache
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error


class SuffixAutomaton:
    """Online suffix automaton over token ids.

    Built incrementally (``extend`` per committed token), it answers the
    retrieval query in O(suffix-link chain) per call: *what is the longest
    suffix of the current sequence that also occurs ending at some earlier
    position, and where does that earlier occurrence end?*

    Each state stores ``first_end`` — the end position (0-based, inclusive)
    of the FIRST occurrence of the substrings it represents. ``first_end``
    is fixed at state creation (clones inherit the original's, since a
    clone's endpos is a superset of the original's and new positions are
    always larger), so no propagation pass is ever needed.

    Pure Python on purpose: per-token construction cost is a few
    microseconds, invisible next to ~76 ms target-model steps.
    """

    __slots__ = ("seq", "_len", "_link", "_next", "_first_end", "_last")

    def __init__(self, tokens: Sequence[int] = ()):
        # State 0 is the root (empty string).
        self.seq: List[int] = []
        self._len = [0]
        self._link = [-1]
        self._next: List[dict] = [{}]
        self._first_end = [-1]
        self._last = 0
        for t in tokens:
            self.extend(t)

    def __len__(self) -> int:
        return len(self.seq)

    def extend(self, token: int) -> None:
        """Append one token to the indexed sequence."""
        token = int(token)
        pos = len(self.seq)
        self.seq.append(token)

        lens, link, nxt, first_end = self._len, self._link, self._next, self._first_end
        cur = len(lens)
        lens.append(lens[self._last] + 1)
        link.append(-1)
        nxt.append({})
        first_end.append(pos)

        p = self._last
        while p != -1 and token not in nxt[p]:
            nxt[p][token] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][token]
            if lens[p] + 1 == lens[q]:
                link[cur] = q
            else:
                clone = len(lens)
                lens.append(lens[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                first_end.append(first_end[q])  # endpos(clone) ⊇ endpos(q)
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def longest_suffix_match(self, max_len: int = 16) -> Tuple[int, int]:
        """Longest suffix of the current sequence with an earlier occurrence.

        Returns ``(match_len, next_pos)`` where ``match_len`` is the length of
        the longest suffix (capped at ``max_len``) that also occurs ending
        strictly before the current end, and ``next_pos`` is the index right
        after that earlier occurrence — i.e. ``seq[next_pos:]`` is the
        retrieval continuation. Returns ``(0, -1)`` when no suffix repeats.

        The earlier occurrence used is the FIRST one in the sequence. Walks
        the suffix-link chain from the last state: endpos sets only grow going
        up the chain, so the deepest state whose ``first_end`` precedes the
        current end holds the longest repeated suffix.
        """
        n = len(self.seq)
        if n < 2:
            return 0, -1
        v = self._last
        while v != 0 and self._first_end[v] >= n - 1:
            v = self._link[v]
        if v == 0:
            return 0, -1
        match_len = min(self._len[v], max_len)
        return match_len, self._first_end[v] + 1


def _require_hybrid_stats(stats) -> None:
    """F7' guard: hybrid/MTP/adaptive paths write draft_*/external_cache_*
    fields that only hybrid_speculative.HybridStats carries. Reject a base
    prompt_lookup stats object at ENTRY with a clear error instead of an
    AttributeError mid-generation."""
    if not hasattr(stats, "draft_proposed"):
        raise TypeError(
            "this generator needs hybrid_speculative.HybridStats (with "
            "draft_*/external_cache_* fields); got a stats object without "
            "them - likely prompt_lookup.HybridStats (F7' footgun)."
        )


@dataclass
class HybridStats(_PromptLookupStatsBase):
    """Per-source accounting for one hybrid generation run.

    Extends prompt_lookup.HybridStats (the F7' unification: shared retrieval/
    plain/span fields are defined ONCE, there) with the draft-chain and
    external-cache accounting the MTP paths need. isinstance-compatible with
    the base, so a hybrid stats object can be passed anywhere the
    prompt-lookup one is expected; the reverse (base into an MTP path) is
    rejected early with a clear error instead of an attribute crash mid-run.
    """

    draft_cycles: int = 0  # cycles whose proposal came from the draft chain
    draft_proposed: int = 0  # tokens proposed by the draft chain
    draft_accepted: int = 0  # ... of which the target accepted
    external_cache_reconciled: bool = False
    external_cache_trimmed_tokens: int = 0
    # MTP rate gate (opt-in): one inline plain probe vs the measured spec
    # rate, then a one-way keep-or-de-latch decision.
    rate_gate_probed: bool = False
    rate_gate_delatched: bool = False
    rate_gate_spec_ms_per_tok: float = 0.0
    rate_gate_plain_ms_per_tok: float = 0.0
    router_plain_cycles: int = 0
    router_reengagements: int = 0
    router_last_num_draft: int = 0
    router_accept_prob: float = 0.0

    @property
    def total_emitted(self) -> int:
        return (
            self.retrieval_accepted
            + self.draft_accepted
            + self.bonus_tokens
            + self.plain_tokens
        )

    @property
    def mean_retrieval_span_proposed(self) -> float:
        return self.retrieval_proposed / max(self.retrieval_cycles, 1)

    @property
    def mean_retrieval_span_accepted(self) -> float:
        return self.retrieval_accepted / max(self.retrieval_cycles, 1)

    @property
    def mean_draft_span_proposed(self) -> float:
        return self.draft_proposed / max(self.draft_cycles, 1)

    @property
    def mean_draft_span_accepted(self) -> float:
        return self.draft_accepted / max(self.draft_cycles, 1)

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        lines = [
            f"cycles: {self.cycles} "
            f"(retrieval {self.retrieval_cycles}, draft {self.draft_cycles}, "
            f"plain {self.plain_cycles})",
            f"tokens: {self.total_emitted} = "
            f"retrieval {self.retrieval_accepted} ({self.retrieval_accepted / tot:.1%})"
            f" + draft {self.draft_accepted} ({self.draft_accepted / tot:.1%})"
            f" + bonus {self.bonus_tokens} ({self.bonus_tokens / tot:.1%})"
            f" + plain {self.plain_tokens} ({self.plain_tokens / tot:.1%})",
        ]
        if self.retrieval_cycles:
            lines.append(
                f"retrieval span: proposed {self.mean_retrieval_span_proposed:.2f} / "
                f"accepted {self.mean_retrieval_span_accepted:.2f} "
                f"(acceptance {self.retrieval_accepted / max(self.retrieval_proposed, 1):.1%})"
            )
        if self.draft_cycles:
            lines.append(
                f"draft span:     proposed {self.mean_draft_span_proposed:.2f} / "
                f"accepted {self.mean_draft_span_accepted:.2f} "
                f"(acceptance {self.draft_accepted / max(self.draft_proposed, 1):.1%})"
            )
        return "\n".join(lines)


def hybrid_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: Optional[nn.Module] = None,
    *,
    num_draft_tokens: int = 3,
    tau: float = 0.55,
    min_match: int = 3,
    max_span: int = 10,
    max_lookback: int = 16,
    max_tokens: int = 256,
    sampler: Optional[Any] = None,
    logits_processors: Optional[Any] = None,
    draft_tokenizer: Optional[Any] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    stats: Optional[HybridStats] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """A generator producing token ids with hybrid (retrieval-first +
    confidence-gated draft chain) speculative decoding. Greedy only.

    Args:
        prompt (mx.array): The input prompt token ids.
        model (nn.Module): The target model.
        draft_model (nn.Module, optional): The draft model (same tokenizer as
          the target). ``None`` gives retrieval-only mode (pure PLD backed by
          the suffix automaton).
        num_draft_tokens (int): Max draft-chain length ``k`` per cycle.
        tau (float): Draft confidence gate — drafting stops at the first token
          whose draft probability is below ``tau``.
        min_match (int): Minimum suffix-match length before retrieval proposes.
        max_span (int): Maximum retrieval continuation span per cycle.
        max_lookback (int): Cap on the suffix-match length considered.
        max_tokens (int): Maximum number of tokens to generate.
        prefill_step_size (int): Chunk size for prompt prefill.
        stats (HybridStats, optional): Updated in place with per-source
          accounting; pass one in to read it after (or during) generation.

    ``sampler``, ``logits_processors``, ``draft_tokenizer`` and
    ``prompt_cache`` are accepted for signature compatibility but must be
    left at their defaults: the hybrid path is greedy-only, same-tokenizer
    only, and manages its own plain ``KVCache`` lists.

    Yields:
        Tuple[int, mx.array, bool]: One committed token, a 1-D vector of log
        probabilities over the vocabulary, and whether the token came from an
        accepted proposal (retrieval or draft) — the same contract as
        ``speculative_generate_step`` (``from_draft`` is ``False`` for bonus/
        correction tokens and plain steps).
    """
    if sampler is not None:
        raise ValueError("hybrid speculative decoding is greedy-only; do not pass a sampler")
    if logits_processors:
        raise ValueError("hybrid speculative decoding does not support logits_processors")
    if draft_tokenizer is not None:
        raise ValueError(
            "hybrid speculative decoding requires target and draft to share a tokenizer"
        )
    if prompt_cache is not None:
        raise NotImplementedError(
            "hybrid speculative decoding manages its own KVCache; "
            "external prompt_cache is not supported"
        )
    if num_draft_tokens < 1:
        raise ValueError("num_draft_tokens must be >= 1")
    if min_match < 1:
        raise ValueError("min_match must be >= 1")
    if not (1 <= max_span):
        raise ValueError("max_span must be >= 1")

    stats = stats if stats is not None else HybridStats()
    _require_hybrid_stats(stats)

    if max_tokens <= 0:
        # Zero-token budget: yield nothing and do no prefill or sampling work.
        return

    y = prompt.astype(mx.uint32)
    # Use each model's own cache layout so hybrid architectures get the right
    # caches (e.g. qwen3_next GatedDeltaNet layers -> ArraysCache, full-attn ->
    # KVCache), not a uniform plain KVCache. Rollback-capable caches make the
    # per-cycle speculative trim exact even for recurrent (non-trimmable) state.
    model_cache = make_prompt_cache(model)
    use_draft = draft_model is not None
    draft_cache = make_prompt_cache(draft_model) if use_draft else None

    # Committed sequence (prompt + everything yielded) and its automaton.
    history: List[int] = [int(t) for t in prompt.tolist()]
    sam = SuffixAutomaton(history)

    def _prefill(m, c, toks):
        # Leave exactly one token unprocessed (mirrors speculative_generate_step's
        # prefill) so the first verify window is a single token + proposal, not the
        # whole prompt tail — keeps the incremental-cache numerics close to plain
        # decode instead of drifting off a large first forward.
        while toks.size > 1:
            n = min(prefill_step_size, toks.size - 1)
            m(toks[:n][None], cache=c)
            mx.eval([layer.state for layer in c])
            toks = toks[n:]
            mx.clear_cache()
        return toks

    def _draft_chain(pending: List[int], k: int) -> Tuple[List[int], int]:
        """Run the confidence-gated draft chain.

        Feeds ``pending`` (committed tokens the draft cache has not seen yet)
        then drafts greedily, stopping at the first token whose probability is
        below ``tau``. Returns ``(proposal, n_fed)`` where ``n_fed`` is how
        many PROPOSED tokens were fed into the draft cache (the caller must
        trim the fed-but-rejected ones after verification).
        """
        proposal: List[int] = []
        n_fed = 0
        feed = mx.array(pending, mx.uint32)
        with mx.stream(generation_stream):
            for _ in range(k):
                logits = draft_model(feed[None], cache=draft_cache)[0, -1, :]
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                tok = mx.argmax(logprobs)
                lp = logprobs[tok]
                mx.eval(tok, lp)
                if math.exp(lp.item()) < tau:
                    break
                proposal.append(int(tok.item()))
                if len(proposal) == k:
                    break  # never feed the last drafted token
                feed = mx.array(proposal[-1:], mx.uint32)
                n_fed += 1
        return proposal, n_fed

    with mx.stream(generation_stream):
        if use_draft:
            draft_tail = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    # After prefill, let rollback-capable caches (GatedDeltaNet ArraysCache) begin
    # recording so each verify forward can be trimmed exactly on rejection; a
    # no-op for plain KV caches. Done post-prefill so we never stash prompt-sized
    # recurrent state — only the small per-cycle verify windows.
    spec_caches = list(model_cache) + (list(draft_cache) if use_draft else [])
    # Rejected proposals must be trimmable; trim_prompt_cache silently no-ops on a
    # non-trimmable cache, which would leave rejected tokens committed and corrupt
    # the output. Both target and draft caches may be trimmed, so validate both.
    _start_speculation_or_cleanup(
        spec_caches,
        spec_caches,
        (
            "hybrid speculative decoding requires a trimmable prompt cache "
            "(recurrent layers need supports_speculative_rollback)."
        ),
    )

    # Tokens committed to `history` but not yet in each model's KV cache.
    pending_target: List[int] = [int(t) for t in y.tolist()]
    pending_draft: List[int] = [int(t) for t in draft_tail.tolist()] if use_draft else []

    ntoks = 0
    try:
        while ntoks < max_tokens:
            stats.cycles += 1
            remaining = max_tokens - ntoks

            # ---- 1. choose a proposal --------------------------------------
            proposal: List[int] = []
            source = "plain"
            n_fed_draft = 0
            budget = remaining - 1  # the bonus token always fills the last slot
            if budget > 0:
                mlen, nxt = sam.longest_suffix_match(max_lookback)
                if mlen >= min_match and 0 <= nxt < len(history):
                    proposal = history[nxt : nxt + min(max_span, budget)]
                    source = "retrieval"
                elif use_draft:
                    proposal, n_fed_draft = _draft_chain(
                        pending_draft, min(num_draft_tokens, budget)
                    )
                    pending_draft = []
                    if proposal:
                        source = "draft"

            # ---- 2. verify: one target forward over [pending, proposal] ----
            n_prop = len(proposal)
            y_verify = mx.array(pending_target + proposal, mx.uint32)
            with mx.stream(generation_stream):
                logits = model(y_verify[None], cache=model_cache)
                rel = logits[0, -(n_prop + 1) :, :]
                logprobs = rel - mx.logsumexp(rel, axis=-1, keepdims=True)
                choices = mx.argmax(logprobs, axis=-1)
            mx.eval(choices)
            choices = choices.tolist()

            n_accept = 0
            while n_accept < n_prop and choices[n_accept] == proposal[n_accept]:
                n_accept += 1
            bonus = choices[n_accept]

            # ---- 3. bookkeeping BEFORE yielding (safe to close at any yield)
            # Target cache: drop the rejected proposal rows. For GatedDeltaNet
            # ArraysCache layers this applies the recorded exact rollback; for
            # KV layers it is the usual offset trim.
            trim_prompt_cache(model_cache, n_prop - n_accept)
            # Draft cache: drop fed-but-rejected draft rows; queue the committed
            # tokens it has not seen for the next draft run.
            if use_draft:
                if source == "draft":
                    trim_prompt_cache(draft_cache, max(n_fed_draft - n_accept, 0))
                    pending_draft = proposal[min(n_fed_draft, n_accept) : n_accept] + [bonus]
                else:
                    # Draft cache untouched this cycle (or consumed only committed
                    # tokens); everything newly committed rides in pending_draft.
                    pending_draft = pending_draft + proposal[:n_accept] + [bonus]
            emitted = proposal[:n_accept] + [bonus]
            history.extend(emitted)
            for t in emitted:
                sam.extend(t)
            pending_target = [bonus]

            if source == "retrieval":
                stats.retrieval_cycles += 1
                stats.retrieval_proposed += n_prop
            elif source == "draft":
                stats.draft_cycles += 1
                stats.draft_proposed += n_prop
            else:
                stats.plain_cycles += 1

            # ---- 4. yield ----------------------------------------------------
            # Delivered-token telemetry is updated exactly at each yield
            # boundary: the consumer may close the generator at any yield
            # (e.g. on EOS), and eager batch accounting would overstate the
            # accepted/bonus/plain token counts.
            for i in range(n_accept):
                ntoks += 1
                if source == "retrieval":
                    stats.retrieval_accepted += 1
                else:
                    stats.draft_accepted += 1
                yield proposal[i], logprobs[i], True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                if source == "plain":
                    stats.plain_tokens += 1
                else:
                    stats.bonus_tokens += 1
                yield bonus, logprobs[n_accept], False
    finally:
        # Stop recording and free rollback stashes on normal completion or when
        # the consumer closes the generator early (e.g. on eos). Every cache's
        # hook runs even if one raises.
        _stop_all_speculation(spec_caches)


def adaptive_pld_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    min_match: int = 3,
    max_span: int = 16,
    cliff_aware_span: bool = False,
    max_lookback: int = 32,
    warmup: int = 48,
    gate: float = 0.12,
    mtp_tail: bool = False,
    num_draft: int = 1,
    persistent_mtp: bool = False,
    mtp_rate_gate: bool = False,
    speculation_router: Optional[RoutedSpeculationPolicy] = None,
    prefill_step_size: int = 512,
    stats: Optional[HybridStats] = None,
    prompt_cache: Optional[Any] = None,
    history_prompt: Optional[mx.array] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Retrieval-only PLD with a one-way latch so it does not lose on no-copy
    output.

    Runs the suffix-automaton retrieval-verify cycle (the ~2x copy-heavy win).
    After ``warmup`` tokens, if the fraction of output that came from retrieval
    is below ``gate``, the work is not copy-heavy, so it **latches once** to a
    tail for all remaining tokens — with no mid-stream thrashing. Copy-heavy work
    never latches, keeping the full PLD speedup with zero regression vs pure PLD.

    The latched tail is a single plain ``generate_step`` (bit-exact, full
    baseline throughput) unless ``mtp_tail=True`` and the model has an MTP head —
    then it self-speculates with the head (``self_mtp_generate_step`` tail),
    giving ~1.1x on the novel code PLD can't help with. So one greedy path serves
    both regimes: copy-heavy -> PLD, novel -> MTP.

    ``persistent_mtp=True`` (with ``mtp_tail``) keeps the MTP tail's KV cache in
    sync with the full committed sequence — the drafting regime vendor-trained
    heads need (see ``self_mtp_generate_step``). The PLD phase routes its verify
    forwards through ``model.model``/``model.logits`` (bit-identical values) so
    the trunk hiddens of every committed token are available, and lazily
    accumulates the (hidden, next_token) pairs; they are teacher-forced into the
    MTP cache in batches (and finally at the latch handoff), so copy-heavy work
    that never latches pays no extra MTP forwards until a flush. Incompatible
    with an external ``prompt_cache``: the cached prefix's hiddens don't exist,
    and an MTP cache missing those positions drafts at wrong RoPE offsets — the
    exact failure persistence exists to fix — so that combination raises.

    ``mtp_rate_gate=True`` protects the MTP tail with a one-shot measured
    break-even check (see ``_mtp_draft_verify_loop``): after a few cycles it
    probes plain decode inline and de-latches the tail permanently if
    speculating isn't actually faster — the tail's acceptance and verify cost
    are workload- and context-dependent, so a config that wins on code
    generation can lose on prose or at long context.

    Greedy only, draft-free; one shared prompt cache. ``prompt_cache`` may hold
    a prefilled prefix; when it is provided, ``prompt`` is the uncached tail and
    ``history_prompt`` must be the full prompt used to seed PLD retrieval history.
    Like all speculative
    decoders, output matches the target's own (batched) greedy — not bit-identical
    to sequential ``generate_step``, since batched/incremental-cache verify forwards
    differ numerically from single-token decode (this holds for upstream
    ``speculative_generate_step`` too). The plain tail runs ``generate_step``
    directly.

    Yields ``(token, logprobs, from_retrieval)``.
    """
    stats = stats if stats is not None else HybridStats()
    _require_hybrid_stats(stats)

    if max_tokens <= 0:
        # Zero-token budget: yield nothing and do no prefill or sampling work
        # (an external prompt_cache is left untouched).
        return

    external_prompt_cache = prompt_cache is not None
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)

    persistent = (
        persistent_mtp and mtp_tail and getattr(model, "mtp", None) is not None
    )
    if persistent and external_prompt_cache:
        raise ValueError(
            "persistent_mtp is incompatible with an external prompt_cache: the "
            "cached prefix's trunk hiddens are unavailable, so the MTP cache "
            "would draft at wrong RoPE offsets. Pass the full prompt instead."
        )
    mtp_cache = model.make_mtp_cache() if persistent else None
    # Committed (hidden, next_token) pairs not yet teacher-forced into
    # mtp_cache; flushed in batches so PLD cycles stay MTP-forward-free.
    mtp_p_hs: List[mx.array] = []
    mtp_p_ts: List[int] = []
    MTP_FLUSH = 256

    y = prompt.astype(mx.uint32)
    with mx.stream(generation_stream):
        prev_h = None  # trunk hidden of the previous chunk's last position
        while y.size > 1:  # leave one token for the first verify window
            n = min(prefill_step_size, y.size - 1)
            if persistent:
                # Teacher-force the MTP over pairs (hidden_i, token_{i+1}) so
                # its KV covers the prompt with real positions (same protocol
                # as self_mtp_generate_step's prefill).
                h_chunk = model.model(y[:n][None], cache=cache)
                if prev_h is None:
                    hs, ts = h_chunk[:, :-1], y[1:n][None]
                else:
                    hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
                    ts = y[:n][None]
                if ts.size > 0:
                    model.mtp_step(hs, ts, mtp_cache)
                prev_h = h_chunk[:, -1:, :]
                mx.eval([c.state for c in mtp_cache])
            else:
                model(y[:n][None], cache=cache)
            mx.eval([c.state for c in cache])
            y = y[n:]
            mx.clear_cache()
        if persistent and prev_h is not None:
            # Pair (h_{L-2}, t_{L-1}); t_{L-1} is the pending verify token.
            model.mtp_step(prev_h, y[None], mtp_cache)
    _start_speculation_or_cleanup(
        cache,
        cache,
        (
            "adaptive PLD requires a trimmable prompt cache "
            "(recurrent layers need supports_speculative_rollback)."
        ),
    )

    history_src = history_prompt if history_prompt is not None else prompt
    history: List[int] = [int(t) for t in history_src.tolist()]
    sam = SuffixAutomaton(history)
    pending: List[int] = [int(t) for t in y.tolist()]  # committed, not yet cached

    ntoks = 0
    retrieved = 0  # tokens emitted from accepted retrieval spans
    latched = False
    cached_unyielded = 0
    try:
        # ---- PLD phase: retrieval-verify cycles until latch or done ----------
        while ntoks < max_tokens and not latched:
            stats.cycles += 1
            budget = (max_tokens - ntoks) - 1  # bonus fills the last slot
            proposal: List[int] = []
            if budget > 0:
                mlen, nxt = sam.longest_suffix_match(max_lookback)
                if mlen >= min_match and 0 <= nxt < len(history):
                    nominal_span = min(max_span, budget)
                    available_span = min(len(history) - nxt, budget)
                    chosen_span = min(nominal_span, available_span)
                    if cliff_aware_span:
                        chosen_span = plan_proposal_around_verify_cliff(
                            nominal_span, available_span, len(pending)
                        )
                        if chosen_span < min(nominal_span, available_span):
                            stats.span_snap_cycles += 1
                            stats.span_snap_tokens += (
                                min(nominal_span, available_span) - chosen_span
                            )
                        elif chosen_span > nominal_span:
                            stats.span_extend_cycles += 1
                            stats.span_extend_tokens += chosen_span - nominal_span
                    proposal = history[nxt : nxt + chosen_span]
            n_prop = len(proposal)

            y_verify = mx.array(pending + proposal, mx.uint32)
            verify_rows = len(pending) + n_prop
            stats.verify_span_hist[verify_rows] = (
                stats.verify_span_hist.get(verify_rows, 0) + 1
            )
            with mx.stream(generation_stream):
                if persistent:
                    # Same values as model(...) — logits = lm_head(model.model)
                    # — but the hiddens stay visible for MTP teacher-forcing.
                    vhidden = model.model(y_verify[None], cache=cache)
                    logits = model.logits(vhidden)
                else:
                    logits = model(y_verify[None], cache=cache)
                rel = logits[0, -(n_prop + 1) :, :]
                logprobs = rel - mx.logsumexp(rel, axis=-1, keepdims=True)
                choices = mx.argmax(logprobs, axis=-1)
            mx.eval(choices)
            choices = choices.tolist()

            n_accept = 0
            while n_accept < n_prop and choices[n_accept] == proposal[n_accept]:
                n_accept += 1
            bonus = choices[n_accept]

            trim_prompt_cache(cache, n_prop - n_accept)
            # Accepted proposal tokens are already in the cache before they are
            # yielded. If the caller closes early, trim any accepted tokens that
            # never reached the caller so an external cache is not over-advanced.
            cached_unyielded = n_accept
            emitted = proposal[:n_accept] + [bonus]
            if persistent:
                # Predecessor hiddens of the committed span: rows for
                # pending[-1] (predicts emitted[0]) through the last accepted
                # proposal token (predicts the bonus). Rejected rows are
                # excluded, so only committed pairs ever reach the MTP cache.
                pre = len(pending) - 1
                mtp_p_hs.append(vhidden[:, pre : pre + n_accept + 1, :])
                mtp_p_ts.extend(emitted)
                if len(mtp_p_ts) >= MTP_FLUSH:
                    with mx.stream(generation_stream):
                        model.mtp_step(
                            mx.concatenate(mtp_p_hs, axis=1),
                            mx.array([mtp_p_ts], mx.uint32),
                            mtp_cache,
                        )
                        mx.eval([c.state for c in mtp_cache])
                    mtp_p_hs, mtp_p_ts = [], []
            history.extend(emitted)
            for t in emitted:
                sam.extend(t)
            pending = [bonus]
            retrieved += n_accept

            if n_prop:
                stats.retrieval_cycles += 1
                stats.retrieval_proposed += n_prop
            else:
                stats.plain_cycles += 1

            # Delivered-token telemetry updates exactly at each yield boundary
            # so an early close (e.g. EOS) never overstates accepted/bonus
            # counts (see the same pattern in prompt_lookup_generate_step).
            for i in range(n_accept):
                ntoks += 1
                cached_unyielded -= 1
                stats.retrieval_accepted += 1
                yield proposal[i], logprobs[i], True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                if n_prop:
                    stats.bonus_tokens += 1
                else:
                    stats.plain_tokens += 1
                yield bonus, logprobs[n_accept], False

            # Latch to plain once we have enough evidence the work isn't copy-heavy.
            if ntoks >= warmup and retrieved / ntoks < gate:
                latched = True

        # ---- latched tail: self-MTP if available+requested, else plain -------
        if latched and ntoks < max_tokens:
            if mtp_tail and getattr(model, "mtp", None) is not None:
                # Keep speculation ON (MTP verify trims on reject). Bootstrap the
                # seed hidden by forwarding the pending token through the trunk.
                with mx.stream(generation_stream):
                    if persistent and mtp_p_ts:
                        # Bring the MTP cache current: pairs end at (·, bonus);
                        # the loop's first draft call then appends the
                        # (h(bonus), nxt) pair with correct positions.
                        model.mtp_step(
                            mx.concatenate(mtp_p_hs, axis=1),
                            mx.array([mtp_p_ts], mx.uint32),
                            mtp_cache,
                        )
                        mtp_p_hs, mtp_p_ts = [], []
                    bh = model.model(mx.array(pending, mx.uint32)[None], cache=cache)
                    blp = model.logits(bh)[0, -1]
                    blp = blp - mx.logsumexp(blp)
                    nxt = int(mx.argmax(blp).item())
                ntoks += 1
                stats.plain_tokens += 1
                yield nxt, blp, False
                yield from _mtp_draft_verify_loop(
                    model,
                    cache,
                    nxt,
                    bh[:, -1:, :],
                    ntoks,
                    max_tokens,
                    num_draft,
                    stats,
                    mtp_cache=mtp_cache,
                    rate_gate=mtp_rate_gate,
                    speculation_router=speculation_router,
                )
            else:
                _stop_all_speculation(cache)
                for tok, lp in generate_step(
                    mx.array(pending, mx.uint32), model,
                    max_tokens=max_tokens - ntoks, prompt_cache=cache, sampler=_GREEDY,
                ):
                    it = int(tok)
                    sam.extend(it)
                    pending = [it]
                    stats.cycles += 1
                    stats.plain_cycles += 1
                    stats.plain_tokens += 1
                    ntoks += 1
                    yield it, lp, False
                    if ntoks == max_tokens:
                        break
    finally:
        try:
            if cached_unyielded > 0:
                trimmed = trim_prompt_cache(cache, cached_unyielded)
                if external_prompt_cache:
                    stats.external_cache_reconciled = True
                    stats.external_cache_trimmed_tokens += int(trimmed or 0)
        finally:
            _stop_all_speculation(cache)


def self_mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    num_draft: int = 1,
    max_tokens: int = 256,
    prefill_step_size: int = 512,
    sampling_temp: float = 0.0,
    persistent_mtp: bool = False,
    rate_gate: bool = False,
    speculation_router: Optional[RoutedSpeculationPolicy] = None,
    stats: Optional[HybridStats] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Self-speculative decoding with the model's own MTP (nextn) head.

    The head drafts the next ``num_draft`` tokens from the trunk's last hidden
    state (no external draft model); the trunk verifies them in one batched
    forward and accepts the greedy-matching prefix. Requires ``model.mtp`` +
    ``mtp_step``/``make_mtp_cache``/``logits`` and (for the hybrid trunk) the GDN
    ``record_rollback`` protocol, so verify steps trim exactly on rejection.
    The head is depth-1, so ``num_draft=1`` is the trained regime; k>1 chains the
    head on its own hidden (out of training distribution — acceptance decays).

    Greedy by default. When ``sampling_temp > 0``, the MTP path uses standard
    speculative rejection sampling with temperature-scaled target and draft
    distributions. This is exact for temperature-only sampling; top-p/top-k need
    a shared distribution transform before they can be made exact here.

    ``persistent_mtp=True`` keeps ONE MTP KV cache in sync with the committed
    sequence (teacher-forced from trunk hiddens during prefill and after each
    verify) instead of a fresh empty cache per draft cycle. The head then
    drafts with full context and real RoPE positions — the regime it was
    trained in — which can lift acceptance dramatically (Hy3-REAP50: 33% ->
    ~80% on code). Costs one extra (single-layer) MTP forward per cycle plus
    ~1 layer-equivalent of prefill; requires the MTP cache to be trimmable.

    ``rate_gate=True`` adds the one-shot measured break-even check from
    ``_mtp_draft_verify_loop``: keep speculating only if it is actually faster
    than an inline plain-decode probe; otherwise fall back to plain for the
    rest of the generation.

    Yields ``(token, logprobs, from_draft)``.
    """
    if getattr(model, "mtp", None) is None:
        raise ValueError("model has no MTP head (build with mtp_num_hidden_layers>0)")
    stats = stats if stats is not None else HybridStats()
    _require_hybrid_stats(stats)

    if max_tokens <= 0:
        # Zero-token budget: yield nothing and do no prefill or sampling work.
        return

    cache = make_prompt_cache(model)
    mtp_cache = model.make_mtp_cache() if persistent_mtp else None

    y = prompt.astype(mx.uint32)
    with mx.stream(generation_stream):
        prev_h = None  # trunk hidden of the previous chunk's last position
        while y.size > 1:  # leave one token to produce the seed hidden
            n = min(prefill_step_size, y.size - 1)
            h_chunk = model.model(y[:n][None], cache=cache)
            if persistent_mtp:
                # Teacher-force the MTP over pairs (hidden_i, token_{i+1}) so
                # its KV covers the prompt with real positions.
                if prev_h is None:
                    hs, ts = h_chunk[:, :-1], y[1:n][None]
                else:
                    hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
                    ts = y[:n][None]
                if ts.size > 0:
                    model.mtp_step(hs, ts, mtp_cache)
                prev_h = h_chunk[:, -1:, :]
                mx.eval([c.state for c in mtp_cache])
            mx.eval([c.state for c in cache])
            y = y[n:]
            mx.clear_cache()
        if persistent_mtp and prev_h is not None:
            model.mtp_step(prev_h, y[None], mtp_cache)  # pair (h_{L-2}, t_{L-1})
        hidden = model.model(y[None], cache=cache)   # [1, 1, H] post-final-norm
        seed_h = hidden[:, -1:, :]                    # trunk hidden at last prompt pos
        first_lp = _temperature_logprobs(model.logits(seed_h)[0, -1], sampling_temp)
        cur = _sample_from_logprobs(first_lp, sampling_temp)
    _start_speculation_or_cleanup(
        cache,
        cache,
        (
            "self-MTP decoding requires a trimmable prompt cache "
            "(recurrent layers need supports_speculative_rollback)."
        ),
    )

    try:
        # Count the token at its yield boundary (not after): an immediate
        # close must still account for the delivered first token.
        stats.plain_tokens += 1
        yield cur, first_lp, False
        yield from _mtp_draft_verify_loop(
            model,
            cache,
            cur,
            seed_h,
            1,
            max_tokens,
            num_draft,
            stats,
            sampling_temp,
            mtp_cache=mtp_cache,
            rate_gate=rate_gate,
            speculation_router=speculation_router,
        )
    finally:
        _stop_all_speculation(cache)


def _temperature_logprobs(logits, sampling_temp: float = 0.0):
    logits = logits.astype(mx.float32)
    if sampling_temp and sampling_temp > 0:
        logits = logits / float(sampling_temp)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _sample_from_logprobs(logprobs, sampling_temp: float = 0.0) -> int:
    if sampling_temp and sampling_temp > 0:
        return int(mx.random.categorical(logprobs).item())
    return int(mx.argmax(logprobs).item())


def _residual_sample(target_logprobs, draft_logprobs, sampling_temp: float) -> int:
    residual = mx.maximum(mx.exp(target_logprobs) - mx.exp(draft_logprobs), 0.0)
    total = mx.sum(residual)
    mx.eval(total)
    if float(total.item()) <= 0.0:
        return _sample_from_logprobs(target_logprobs, sampling_temp)
    residual_logprobs = mx.log(residual / total)
    return int(mx.random.categorical(residual_logprobs).item())


def _accept_sampled_draft(target_logprobs, draft_logprobs, token: int) -> bool:
    target_p = mx.exp(target_logprobs[token])
    draft_p = mx.exp(draft_logprobs[token])
    ratio = mx.minimum(1.0, target_p / mx.maximum(draft_p, 1e-30))
    u = mx.random.uniform(shape=())
    mx.eval(ratio, u)
    return float(u.item()) <= float(ratio.item())


def _mtp_draft_verify_loop(
    model,
    cache,
    cur,
    seed_h,
    ntoks,
    max_tokens,
    num_draft,
    stats,
    sampling_temp: float = 0.0,
    mtp_cache=None,
    rate_gate: bool = False,
    speculation_router: Optional[RoutedSpeculationPolicy] = None,
):
    """Shared MTP tail: the head drafts, the trunk verifies, GDN rollback trims
    rejects. Assumes cache speculation is already ON; ``cur`` is the last
    committed-but-uncached token and ``seed_h`` the trunk hidden that predicted
    it. Yields (token, logprobs, from_draft). Does NOT start/stop speculation.

    When ``mtp_cache`` is given it is a PERSISTENT context cache covering the
    committed pairs (hidden_i, token_{i+1}), so the head drafts with full
    context at real RoPE positions (its trained regime). After each verify the
    k draft entries are rewound; the newly committed span (with TRUNK hiddens)
    is carried as ``pending`` pairs and teacher-forced as a prefix of the next
    cycle's first draft call — one MTP forward per cycle, no separate
    catch-up pass. ``None`` keeps the legacy fresh-cache-per-cycle behavior.

    ``rate_gate=True`` adds a one-shot empirical break-even check: after
    ``_RATE_GATE_WARMUP_CYCLES`` measured draft/verify cycles, it decodes
    ``_RATE_GATE_PROBE_TOKENS`` tokens plainly INLINE (the probe tokens are
    delivered output, nothing is wasted), compares wall-clock ms per delivered
    token, and decides ONCE: keep speculating only if the spec rate beats the
    plain rate by ``_RATE_GATE_MARGIN``; otherwise de-latch to plain decode for
    the rest of the generation. Measured, not modeled — the verify-cost
    break-even is target- AND context-dependent (see
    lessons/persistent-mtp-context-cache) — and one-way, so no mid-stream
    thrashing (the adaptive-PLD latch philosophy; per the D-Cut lesson,
    continuous adaptivity loses to simple decisions)."""
    persistent = mtp_cache is not None
    pending_hs = None  # committed (hidden, token) pairs not yet in mtp_cache
    pending_ts: List[int] = []
    gated_off = False
    gate_cycles = 0
    spec_secs = 0.0  # wall-clock over measured spec cycles
    spec_toks = 0  # tokens those cycles delivered
    router_plain = False

    def _plain_step():
        # One width-1 trunk forward: commits `cur`, samples the next token.
        # Keeps the pending-pair protocol intact so persistent drafting can
        # resume seamlessly after a probe.
        nonlocal cur, seed_h, pending_hs, pending_ts
        with mx.stream(generation_stream):
            h = model.model(mx.array([[cur]], mx.uint32), cache=cache)
            lp = _temperature_logprobs(model.logits(h)[0, -1], sampling_temp)
            nxt = _sample_from_logprobs(lp, sampling_temp)
        if persistent and not gated_off:
            # Pairs only matter if drafting can resume; after a permanent
            # de-latch they would just accumulate unused memory.
            pending_hs = (
                seed_h if pending_hs is None
                else mx.concatenate([pending_hs, seed_h], axis=1)
            )
            pending_ts.append(cur)
        seed_h, cur = h[:, -1:, :], nxt
        stats.cycles += 1
        stats.plain_cycles += 1
        stats.plain_tokens += 1
        return nxt, lp

    while ntoks < max_tokens:
        if gated_off:
            tok_, lp = _plain_step()
            ntoks += 1
            yield tok_, lp, False
            continue
        routed_k = None
        if speculation_router is not None:
            decision = speculation_router.decide(
                max_draft=num_draft,
                remaining=max_tokens - ntoks,
            )
            routed_k = decision.num_draft
            stats.router_last_num_draft = routed_k
            stats.router_accept_prob = decision.accept_prob
            stats.router_reengagements = speculation_router.reengagements
            if routed_k == 0:
                if not router_plain:
                    _stop_all_speculation(cache)
                    router_plain = True
                tok_, lp = _plain_step()
                ntoks += 1
                stats.router_plain_cycles += 1
                yield tok_, lp, False
                continue
            if router_plain:
                _start_speculation_or_cleanup(
                    cache,
                    cache,
                    "routed MTP re-entry needs a trimmable prompt cache.",
                )
                router_plain = False
        if rate_gate and not stats.rate_gate_probed and gate_cycles >= _RATE_GATE_WARMUP_CYCLES:
            stats.rate_gate_probed = True
            # Honest plain reference: rollback recording (GDN/rotating-cache
            # speculation bookkeeping) is spec-only overhead, so switch it off
            # for the probe — and leave it off after a de-latch, which is also
            # what makes the plain fallback run at true baseline speed. All
            # tokens are committed at this point, so there is nothing to lose.
            _stop_all_speculation(cache)
            probe_t0 = time.perf_counter()
            n_probe = 0
            while n_probe < _RATE_GATE_PROBE_TOKENS and ntoks < max_tokens:
                tok_, lp = _plain_step()
                ntoks += 1
                n_probe += 1
                yield tok_, lp, False
            plain_rate = (time.perf_counter() - probe_t0) * 1000.0 / max(n_probe, 1)
            spec_rate = spec_secs * 1000.0 / max(spec_toks, 1)
            stats.rate_gate_spec_ms_per_tok = spec_rate
            stats.rate_gate_plain_ms_per_tok = plain_rate
            if spec_rate > plain_rate * (1.0 - _RATE_GATE_MARGIN):
                gated_off = True
                stats.rate_gate_delatched = True
            else:
                _start_speculation_or_cleanup(
                    cache,
                    cache,
                    "MTP rate-gate resume needs a trimmable prompt cache.",
                )
            continue
        cycle_t0 = time.perf_counter()
        ntoks_at_cycle_start = ntoks
        stats.cycles += 1
        k = min(routed_k or num_draft, max_tokens - ntoks)

        # ---- draft k tokens with the MTP head (chained) ----------------------
        if not persistent:
            mtp_cache = model.make_mtp_cache()
        drafts: List[int] = []
        draft_logprobs: List[mx.array] = []
        h, tok = seed_h, mx.array([[cur]], mx.uint32)
        with mx.stream(generation_stream):
            for i in range(k):
                if i == 0 and pending_hs is not None:
                    hs = mx.concatenate([pending_hs, h], axis=1)
                    ts = mx.array([pending_ts + [cur]], mx.uint32)
                else:
                    hs, ts = h, tok
                d_logits, post = model.mtp_step(hs, ts, mtp_cache)
                h = post[:, -1:, :]
                d_lp = _temperature_logprobs(d_logits[0, -1], sampling_temp)
                d = _sample_from_logprobs(d_lp, sampling_temp)
                drafts.append(d)
                draft_logprobs.append(d_lp)
                tok = mx.array([[d]], mx.uint32)

        # ---- verify: trunk over [cur, drafts...] in one forward --------------
        verify_in = mx.array([[cur] + drafts], mx.uint32)   # [1, k+1]
        with mx.stream(generation_stream):
            vhidden = model.model(verify_in, cache=cache)   # [1, k+1, H]
            vlogits = model.logits(vhidden)                 # [1, k+1, V]
            logprobs = _temperature_logprobs(vlogits[0], sampling_temp)
            targets = mx.argmax(logprobs, axis=-1)
        mx.eval(targets, vhidden)

        n_accept = 0
        if sampling_temp and sampling_temp > 0:
            while (
                n_accept < k
                and _accept_sampled_draft(
                    logprobs[n_accept], draft_logprobs[n_accept], drafts[n_accept]
                )
            ):
                n_accept += 1
            if n_accept < k:
                bonus = _residual_sample(
                    logprobs[n_accept], draft_logprobs[n_accept], sampling_temp
                )
            else:
                bonus = _sample_from_logprobs(logprobs[n_accept], sampling_temp)
        else:
            targets = targets.tolist()
            while n_accept < k and targets[n_accept] == drafts[n_accept]:
                n_accept += 1
            bonus = targets[n_accept]

        # Trunk cache advanced by k+1 (cur + k drafts); keep cur + n_accept.
        trim_prompt_cache(cache, k - n_accept)
        if persistent:
            # Rewind the k speculative entries: (h_p, cur) plus the k-1
            # chained pairs built from MTP (not trunk) hiddens. The committed
            # span — (h_p, cur), (h_{p+1}, d_1) .. (h_{p+n_accept}, d_na) with
            # TRUNK hiddens — is carried as pending pairs and re-fed as the
            # prefix of the next cycle's first draft call. The bonus token
            # stays out: it becomes the next cycle's cur.
            trim_prompt_cache(mtp_cache, k)
            if n_accept > 0:
                pending_hs = mx.concatenate(
                    [seed_h, vhidden[:, :n_accept, :]], axis=1
                )
            else:
                pending_hs = seed_h
            pending_ts = [cur] + drafts[:n_accept]
        seed_h = vhidden[:, n_accept : n_accept + 1, :]  # hidden that predicted bonus
        stats.draft_proposed += k
        stats.draft_cycles += 1
        if speculation_router is not None:
            speculation_router.observe(k, n_accept)
            stats.router_accept_prob = speculation_router.accept_prob

        # Delivered-token telemetry updates exactly at each yield boundary so
        # an early close (e.g. EOS) never overstates accepted/bonus counts.
        for i in range(n_accept):
            ntoks += 1
            stats.draft_accepted += 1
            yield drafts[i], logprobs[i], True
            if ntoks == max_tokens:
                break
        if ntoks < max_tokens:
            ntoks += 1
            stats.bonus_tokens += 1
            yield bonus, logprobs[n_accept], False
        cur = bonus
        spec_secs += time.perf_counter() - cycle_t0
        spec_toks += ntoks - ntoks_at_cycle_start
        gate_cycles += 1


def hybrid_stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    draft_model: Optional[nn.Module] = None,
    *,
    max_tokens: int = 256,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """Stream ``GenerationResponse`` objects from hybrid speculative decoding
    — the hybrid twin of ``stream_generate`` with a draft model.

    Args:
        model (nn.Module): The target model.
        tokenizer: The tokenizer (shared by target and draft).
        prompt: The input prompt string or integer tokens.
        draft_model (nn.Module, optional): The draft model; ``None`` for
          retrieval-only (pure PLD) mode.
        max_tokens (int): Maximum number of tokens to generate.
        kwargs: Forwarded to ``hybrid_generate_step`` (``tau``, ``min_match``,
          ``max_span``, ``num_draft_tokens``, ``stats``, ...).
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    token_generator = hybrid_generate_step(
        prompt, model, draft_model, max_tokens=max_tokens, **kwargs
    )
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        n = -1
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

        if n < 0:
            return  # generator yielded nothing (e.g. max_tokens=0): no summary

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )
