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
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .generate import (
    GenerationResponse,
    generate_step,
    generation_stream,
    wired_limit,
)
from .models.cache import KVCache, make_prompt_cache, trim_prompt_cache
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper

_GREEDY = make_sampler(temp=0.0)


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


@dataclass
class HybridStats:
    """Per-source accounting for one hybrid generation run."""

    cycles: int = 0
    retrieval_cycles: int = 0  # cycles whose proposal came from retrieval
    draft_cycles: int = 0  # cycles whose proposal came from the draft chain
    plain_cycles: int = 0  # cycles with no proposal (single target step)
    retrieval_proposed: int = 0  # tokens proposed by retrieval
    retrieval_accepted: int = 0  # ... of which the target accepted
    draft_proposed: int = 0  # tokens proposed by the draft chain
    draft_accepted: int = 0  # ... of which the target accepted
    bonus_tokens: int = 0  # bonus/correction tokens from propose cycles
    plain_tokens: int = 0  # tokens emitted by plain (no-proposal) cycles

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
        while toks.size > prefill_step_size:
            m(toks[:prefill_step_size][None], cache=c)
            mx.eval([layer.state for layer in c])
            toks = toks[prefill_step_size:]
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
    for c in spec_caches:
        c.start_speculation()

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
                stats.retrieval_accepted += n_accept
                stats.bonus_tokens += 1
            elif source == "draft":
                stats.draft_cycles += 1
                stats.draft_proposed += n_prop
                stats.draft_accepted += n_accept
                stats.bonus_tokens += 1
            else:
                stats.plain_cycles += 1
                stats.plain_tokens += 1

            # ---- 4. yield ----------------------------------------------------
            for i in range(n_accept):
                ntoks += 1
                yield proposal[i], logprobs[i], True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                yield bonus, logprobs[n_accept], False
    finally:
        # Stop recording and free rollback stashes on normal completion or when
        # the consumer closes the generator early (e.g. on eos).
        for c in spec_caches:
            c.stop_speculation()


def adaptive_pld_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    min_match: int = 3,
    max_span: int = 16,
    max_lookback: int = 32,
    warmup: int = 48,
    gate: float = 0.12,
    mtp_tail: bool = False,
    num_draft: int = 1,
    prefill_step_size: int = 512,
    stats: Optional[HybridStats] = None,
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

    Greedy only, draft-free; one shared prompt cache. The plain tail is
    single-token forwards (the exact sequential path, bit-exact); only committed
    multi-token retrieval spans carry the model's batched-verify numerics.

    Yields ``(token, logprobs, from_retrieval)``.
    """
    stats = stats if stats is not None else HybridStats()
    cache = make_prompt_cache(model)

    y = prompt.astype(mx.uint32)
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=cache)
            mx.eval([c.state for c in cache])
            y = y[prefill_step_size:]
            mx.clear_cache()
    for c in cache:
        c.start_speculation()

    history: List[int] = [int(t) for t in prompt.tolist()]
    sam = SuffixAutomaton(history)
    pending: List[int] = [int(t) for t in y.tolist()]  # committed, not yet cached

    ntoks = 0
    retrieved = 0  # tokens emitted from accepted retrieval spans
    latched = False
    try:
        # ---- PLD phase: retrieval-verify cycles until latch or done ----------
        while ntoks < max_tokens and not latched:
            stats.cycles += 1
            budget = (max_tokens - ntoks) - 1  # bonus fills the last slot
            proposal: List[int] = []
            if budget > 0:
                mlen, nxt = sam.longest_suffix_match(max_lookback)
                if mlen >= min_match and 0 <= nxt < len(history):
                    proposal = history[nxt : nxt + min(max_span, budget)]
            n_prop = len(proposal)

            y_verify = mx.array(pending + proposal, mx.uint32)
            with mx.stream(generation_stream):
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
            emitted = proposal[:n_accept] + [bonus]
            history.extend(emitted)
            for t in emitted:
                sam.extend(t)
            pending = [bonus]
            retrieved += n_accept

            if n_prop:
                stats.retrieval_cycles += 1
                stats.retrieval_proposed += n_prop
                stats.retrieval_accepted += n_accept
                stats.bonus_tokens += 1
            else:
                stats.plain_cycles += 1
                stats.plain_tokens += 1

            for i in range(n_accept):
                ntoks += 1
                yield proposal[i], logprobs[i], True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
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
                    bh = model.model(mx.array(pending, mx.uint32)[None], cache=cache)
                    blp = model.logits(bh)[0, -1]
                    blp = blp - mx.logsumexp(blp)
                    nxt = int(mx.argmax(blp).item())
                ntoks += 1
                stats.plain_tokens += 1
                yield nxt, blp, False
                yield from _mtp_draft_verify_loop(
                    model, cache, nxt, bh[:, -1:, :], ntoks, max_tokens, num_draft, stats
                )
            else:
                for c in cache:
                    c.stop_speculation()
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
        for c in cache:
            c.stop_speculation()


def self_mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    num_draft: int = 1,
    max_tokens: int = 256,
    prefill_step_size: int = 512,
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

    Greedy only. Yields ``(token, logprobs, from_draft)``.
    """
    if getattr(model, "mtp", None) is None:
        raise ValueError("model has no MTP head (build with mtp_num_hidden_layers>0)")
    stats = stats if stats is not None else HybridStats()
    cache = make_prompt_cache(model)

    y = prompt.astype(mx.uint32)
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model.model(y[:prefill_step_size][None], cache=cache)
            mx.eval([c.state for c in cache])
            y = y[prefill_step_size:]
            mx.clear_cache()
        hidden = model.model(y[None], cache=cache)   # [1, S, H] post-final-norm
        seed_h = hidden[:, -1:, :]                    # trunk hidden at last prompt pos
        first_lp = model.logits(seed_h)[0, -1]
        first_lp = first_lp - mx.logsumexp(first_lp)
        cur = int(mx.argmax(first_lp).item())
    for c in cache:
        c.start_speculation()

    try:
        yield cur, first_lp, False
        stats.plain_tokens += 1
        yield from _mtp_draft_verify_loop(
            model, cache, cur, seed_h, 1, max_tokens, num_draft, stats
        )
    finally:
        for c in cache:
            c.stop_speculation()


def _mtp_draft_verify_loop(model, cache, cur, seed_h, ntoks, max_tokens, num_draft, stats):
    """Shared MTP tail: the head drafts, the trunk verifies, GDN rollback trims
    rejects. Assumes cache speculation is already ON; ``cur`` is the last
    committed-but-uncached token and ``seed_h`` the trunk hidden that predicted
    it. Yields (token, logprobs, from_draft). Does NOT start/stop speculation."""
    while ntoks < max_tokens:
        stats.cycles += 1
        k = min(num_draft, max_tokens - ntoks)

        # ---- draft k tokens with the MTP head (chained) ----------------------
        mtp_cache = model.make_mtp_cache()
        drafts: List[int] = []
        h, tok = seed_h, mx.array([[cur]], mx.uint32)
        with mx.stream(generation_stream):
            for _ in range(k):
                d_logits, h = model.mtp_step(h, tok, mtp_cache)
                d = int(mx.argmax(d_logits[0, -1]).item())
                drafts.append(d)
                tok = mx.array([[d]], mx.uint32)

        # ---- verify: trunk over [cur, drafts...] in one forward --------------
        verify_in = mx.array([[cur] + drafts], mx.uint32)   # [1, k+1]
        with mx.stream(generation_stream):
            vhidden = model.model(verify_in, cache=cache)   # [1, k+1, H]
            vlogits = model.logits(vhidden)                 # [1, k+1, V]
            logprobs = vlogits[0] - mx.logsumexp(vlogits[0], axis=-1, keepdims=True)
            targets = mx.argmax(logprobs, axis=-1)
        mx.eval(targets, vhidden)
        targets = targets.tolist()

        n_accept = 0
        while n_accept < k and targets[n_accept] == drafts[n_accept]:
            n_accept += 1
        bonus = targets[n_accept]

        # Trunk cache advanced by k+1 (cur + k drafts); keep cur + n_accept.
        trim_prompt_cache(cache, k - n_accept)
        seed_h = vhidden[:, n_accept : n_accept + 1, :]  # hidden that predicted bonus
        stats.draft_proposed += k
        stats.draft_accepted += n_accept
        stats.draft_cycles += 1
        stats.bonus_tokens += 1

        for i in range(n_accept):
            ntoks += 1
            yield drafts[i], logprobs[i], True
            if ntoks == max_tokens:
                break
        if ntoks < max_tokens:
            ntoks += 1
            yield bonus, logprobs[n_accept], False
        cur = bonus


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
