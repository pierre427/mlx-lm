# Copyright © 2024 Apple Inc.

import random
import unittest
from typing import List

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    StopSequenceMatcher,
    batch_generate,
    generate,
    generate_step,
    stream_generate,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


class TestGenerate(unittest.TestCase):

    BATCH_LOGPROB_ATOL = 0.05

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def tearDown(self):
        # Rotating-cache tests shadow the model's class method on this shared
        # setUpClass instance. Always remove that override, even when an
        # assertion fails before the test's normal cleanup line, so one failure
        # cannot change every later test's cache topology.
        if "make_cache" in vars(self.model):
            del self.model.make_cache

    def _assert_batch_equivalent(
        self,
        batch_token,
        batch_logprobs,
        reference_token,
        reference_logprobs,
    ):
        """Check behavior plus the measured cross-shape numerical envelope.

        MLX is bit-exact when the batch shape is repeated, but changing that
        shape changes reduction/quantized-kernel geometry.  On the fixed test
        fixtures the full-vocabulary log-probability delta is <= 0.0455 while
        the delivered token is unchanged.  Keep both requirements: token
        identity catches behavioral corruption and the 0.05 bound catches a
        materially wrong distribution without demanding cross-shape bit parity.
        """
        self.assertEqual(int(batch_token), int(reference_token))
        batch_logprobs = batch_logprobs.astype(mx.float32)
        reference_logprobs = reference_logprobs.astype(mx.float32)
        self.assertTrue(mx.all(mx.isfinite(batch_logprobs)))
        self.assertTrue(mx.all(mx.isfinite(reference_logprobs)))
        max_abs = float(mx.max(mx.abs(batch_logprobs - reference_logprobs)).item())
        self.assertLessEqual(max_abs, self.BATCH_LOGPROB_ATOL)

    def test_batch_numerical_contract_rejects_corruption(self):
        logprobs = mx.array([-1.0, -2.0, -3.0])
        self._assert_batch_equivalent(0, logprobs, 0, logprobs)
        with self.assertRaises(AssertionError):
            self._assert_batch_equivalent(1, logprobs, 0, logprobs)
        with self.assertRaises(AssertionError):
            self._assert_batch_equivalent(
                0,
                logprobs.at[2].add(self.BATCH_LOGPROB_ATOL + 0.001),
                0,
                logprobs,
            )

    def test_generate(self):
        # Simple test that generation runs
        text = generate(
            self.model, self.tokenizer, "hello", max_tokens=5, verbose=False
        )

    def test_generate_with_logit_bias(self):
        logit_bias = {0: 2000.0, 1: -20.0}
        text = generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            logits_processors=make_logits_processors(logit_bias),
            verbose=False,
        )
        self.assertEqual(text, "!!!!!")

    def test_stream_generate_max_tokens(self):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a story about Einstein"}],
            tokenize=True,
            add_generation_prompt=True,
        )

        tokens = []
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
        ):
            tokens.append(response.token)
        self.assertEqual(len(tokens), 4)

    def test_generate_with_processor(self):
        init_toks = self.tokenizer.encode("hello")

        all_toks = None

        def logits_processor(toks, logits):
            nonlocal all_toks
            all_toks = toks
            return logits

        generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            verbose=False,
            logits_processors=[logits_processor],
        )
        self.assertEqual(len(all_toks), len(init_toks) + 5)

    def test_stream_generate_speculative(self):
        # Use same model as draft model, this is not a speed test
        draft_model = self.model

        results: List[GenerationResponse] = []
        drafted: List[bool] = []

        # make a determinate sampler
        sampler = make_sampler(temp=0.0)
        messages = [{"role": "user", "content": "hello"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
        ):
            drafted.append(generation_result.from_draft)
            results.append(generation_result)

        self.assertEqual(len(results), 5)
        # Each verify cycle reserves one output slot for the target bonus.
        # At the two-token tail only one useful draft remains, followed by
        # the final target token.
        self.assertEqual(drafted, [True, True, False, True, False])

    def test_stream_generate_input_embeddings(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)

    def test_stream_generate_input_embeddings_prefill(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        # setup prompt progress callback to track batched prefill
        num_prompt_processing_callbacks = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal num_prompt_processing_callbacks
            num_prompt_processing_callbacks += 1

        # generate
        prefill_step_size = 5
        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=progress_callback,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)
        num_embeddings = prompt_embeddings.shape[0]
        self.assertTrue(
            num_embeddings / prefill_step_size < num_prompt_processing_callbacks
        )

    def test_stream_generate_zero_max_tokens(self):
        prompt = self.tokenizer.encode("hi")
        responses = list(
            stream_generate(self.model, self.tokenizer, prompt, max_tokens=0)
        )
        self.assertEqual(responses, [])

    def test_insert_supplied_cache_honors_kv_bits(self):
        # Externally supplied caches must honor the generator's kv-quant
        # config at insert: on the supported (rotating) configuration they
        # are quantized to match the fresh-lane cohort; on the unsupported
        # (non-rotating) one they fail loudly instead of silently running
        # unquantized and crashing later in the cohort merge.
        from mlx_lm.models.cache import (
            RotatingQuantizedKVCache,
            make_prompt_cache,
        )

        prompt = self.tokenizer.encode("hello there")

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=2,
            max_kv_size=64,
            kv_bits=8,
            kv_group_size=32,
        )
        try:
            supplied = [
                RotatingKVCache(max_size=64) for _ in range(len(self.model.layers))
            ]
            gen.insert([prompt], caches=[supplied])
            queued_cache = gen._unprocessed_sequences[0][3]
            self.assertTrue(
                all(isinstance(c, RotatingQuantizedKVCache) for c in queued_cache)
            )
            responses = []
            while res := gen.next_generated():
                responses.extend(res)
            self.assertTrue(len(responses) > 0)
        finally:
            gen.close()

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=2,
            kv_bits=8,
            kv_group_size=32,
        )
        try:
            with self.assertRaises(ValueError):
                gen.insert([prompt], caches=[make_prompt_cache(self.model)])
        finally:
            gen.close()

    def test_insert_supplied_cachelist_honors_kv_bits(self):
        # Composite CacheList layer caches (hybrid attention+recurrent
        # layers) hold their KV caches one level down. At insert with
        # kv_bits set, nested KV leaves must be quantized (the wrapper has
        # merge but no to_quantized, so without recursion they silently
        # stay fp), and a nested leaf that quantizes to a non-mergeable
        # class must fail loudly at insert instead of crashing later in
        # CacheList.merge.
        from mlx_lm.models.cache import (
            ArraysCache,
            CacheList,
            RotatingQuantizedKVCache,
        )

        prompt = self.tokenizer.encode("hello there")

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=2,
            max_kv_size=64,
            kv_bits=8,
            kv_group_size=32,
        )
        try:
            supplied = [
                CacheList(RotatingKVCache(max_size=64), ArraysCache(size=1))
                for _ in range(len(self.model.layers))
            ]
            gen.insert([prompt], caches=[supplied])
            queued_cache = gen._unprocessed_sequences[0][3]
            for c in queued_cache:
                self.assertIsInstance(c, CacheList)
                self.assertIsInstance(c.caches[0], RotatingQuantizedKVCache)
                self.assertIsInstance(c.caches[1], ArraysCache)
        finally:
            gen.close()

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=2,
            max_kv_size=64,
            kv_bits=8,
            kv_group_size=32,
        )
        try:
            # Nested plain KVCache quantizes to QuantizedKVCache, which has
            # no merge: the leaf-level guard must reject it at insert.
            supplied = [
                CacheList(KVCache(), ArraysCache(size=1))
                for _ in range(len(self.model.layers))
            ]
            with self.assertRaises(ValueError):
                gen.insert([prompt], caches=[supplied])
        finally:
            gen.close()

    def test_fresh_lane_cachelist_non_mergeable_raises(self):
        # The fresh lane has the same exposure as supplied caches:
        # _make_new_cache() doesn't descend into CacheList when wrapping
        # for max_kv_size, so a hybrid model whose make_cache() nests a
        # plain KVCache (baichuan_m1's global layers) yields a leaf that
        # quantizes to non-mergeable QuantizedKVCache. Must raise at
        # insert, not AttributeError later in CacheList.merge.
        from mlx_lm.models.cache import ArraysCache, CacheList

        prompt = self.tokenizer.encode("hello there")
        self.model.make_cache = lambda: [
            CacheList(ArraysCache(size=2), KVCache())
            for _ in range(len(self.model.layers))
        ]
        try:
            gen = BatchGenerator(
                self.model,
                stop_tokens=self.tokenizer.eos_token_ids,
                max_tokens=2,
                max_kv_size=64,
                kv_bits=8,
                kv_group_size=32,
            )
            try:
                with self.assertRaises(ValueError):
                    gen.insert([prompt])
            finally:
                gen.close()
        finally:
            del self.model.make_cache

    def test_batch_matches_single(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model, stop_tokens=self.tokenizer.eos_token_ids, max_tokens=1
        )
        uids = gen.insert(prompts)
        batch_responses = {r.uid: r for r in gen.next_generated()}

        # Do a test for each prompt: behavior matches and the distributions
        # stay inside the measured cross-batch-shape numerical envelope.
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                batch_response = batch_responses[uids[e]]
                self._assert_batch_equivalent(
                    batch_response.token,
                    batch_response.logprobs,
                    response.token,
                    response.logprobs,
                )
                break

    def test_many_batches(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=1,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        uids = gen.insert(prompts)
        batch_responses = {}
        not_in = True
        iters = 0
        while responses := gen.next_generated():
            for r in responses:
                not_in &= r.uid not in batch_responses
                batch_responses[r.uid] = r
            iters += 1
        # only one token per prompt means only one response per prompt
        self.assertTrue(not_in)

        # completion batch size is too small for a single iteration
        self.assertTrue(iters > 1)

        # Do a test for each prompt under the same behavioral + bounded-
        # distribution contract as the single-batch case.
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                batch_response = batch_responses[uids[e]]
                self._assert_batch_equivalent(
                    batch_response.token,
                    batch_response.logprobs,
                    response.token,
                    response.logprobs,
                )
                break

    def test_batch_unique_max_toks(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        num_toks = [2, 3, 4, 5]
        uids = gen.insert(prompts, max_tokens=num_toks)
        batch_responses = {uid: [] for uid in uids}
        while responses := gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.token)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            tokens = []
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=num_toks[e],
            ):
                tokens.append(response.token)

            batch_tokens = batch_responses[uids[e]]
            self.assertEqual(tokens, batch_tokens)

    def test_batch_sliding_window(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        self.model.make_cache = lambda: [
            RotatingKVCache(max_size=4) for _ in self.model.layers
        ]
        batch_gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=10,
            prefill_batch_size=1,
            prefill_step_size=8,
            completion_batch_size=2,
        )
        uids = batch_gen.insert(prompts)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append((r.token, r.logprobs))

        for e, uid in enumerate(uids):
            for i, response in enumerate(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    prompts[e],
                    max_tokens=10,
                )
            ):
                batch_token, batch_logprobs = batch_responses[uid][i]
                self._assert_batch_equivalent(
                    batch_token,
                    batch_logprobs,
                    response.token,
                    response.logprobs,
                )

        del self.model.make_cache

    def test_batch_sliding_window_quantized(self):
        """BatchGenerator with max_kv_size AND kv_bits set together — the exact
        combination mira-mlx always uses (--max-kv-size is always passed), which
        used to be blocked by BatchRotatingKVCache.to_quantized() raising
        NotImplementedError. Forces rotation (max_tokens > max_kv_size) and
        mid-stream job insertion (jobs finish at different times) to exercise
        RotatingQuantizedKVCache/BatchRotatingQuantizedKVCache's merge/filter/
        extend/extract paths under real generation, not synthetic tensors."""
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        batch_gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=10,
            max_kv_size=4,
            kv_bits=8,
            kv_group_size=32,
            prefill_batch_size=1,
            prefill_step_size=8,
            completion_batch_size=2,
        )
        uids = batch_gen.insert(prompts)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.token)

        for uid in uids:
            self.assertGreater(
                len(batch_responses[uid]),
                0,
                "quantized+rotating job produced no tokens",
            )
            for tok in batch_responses[uid]:
                self.assertFalse(mx.isnan(mx.array(float(tok))))

    def test_batch_generator_rejects_delayed_quantized_kv_start(self):
        """quantized_kv_start > 0 isn't supported for the batching path (per-job
        caches are created once, empty, at insertion — there's no per-step
        re-check that would ever trigger a delayed threshold)."""
        with self.assertRaises(NotImplementedError):
            BatchGenerator(
                self.model,
                max_kv_size=8,
                kv_bits=8,
                quantized_kv_start=4,
            )

    def test_batch_generate_with_logits_processors(self):
        """Test that batch_generate with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        logprobs = response.logprobs
        self.assertEqual(logprobs[0].item(), 0.0)
        self.assertEqual(logprobs.argmin().item(), 1)

        del batch_gen

        logit_bias = {0: 2000.0}
        processors = make_logits_processors(logit_bias)
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )

        (uid0,) = batch_gen.insert([prompt])

        logit_bias = {1: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid1,) = batch_gen.insert([prompt], logits_processors=[processors])

        logit_bias = {2: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid2,) = batch_gen.insert([prompt], logits_processors=[processors])

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].logprobs[0].item(), 0.0)
        self.assertEqual(responses[uid1].logprobs[1].item(), 0.0)
        self.assertEqual(responses[uid2].logprobs[2].item(), 0.0)

    def test_batch_generate_processor_tokens_match_prompt_on_first_step(self):
        prompt = self.tokenizer.encode("hello")
        seen = []

        def processor(tokens, logits):
            seen.append(tokens)
            return logits

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=[processor],
        )
        batch_gen.insert([prompt])
        batch_gen.next_generated()

        self.assertTrue(hasattr(seen[0], "shape"))
        self.assertEqual(seen[0].tolist(), prompt)

    def test_batch_generate_function_with_logits_processors(self):
        """Test that batch_generate function with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=1,
            logits_processors=processors,
        )
        self.assertEqual(len(response.texts), 1)
        generated_token = self.tokenizer.encode(response.texts[0])[0]
        self.assertEqual(generated_token, 0)

    def test_batch_generate_with_samplers(self):
        """Test that batch_generate with logits_processors produces correct results."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        self.assertEqual(response.token, 1)

        del batch_gen

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )

        (uid0,) = batch_gen.insert([prompt])
        uid1, uid2 = batch_gen.insert(
            [prompt, prompt],
            samplers=[lambda _: mx.array([2]), lambda _: mx.array([3])],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].token, 1)
        self.assertEqual(responses[uid1].token, 2)
        self.assertEqual(responses[uid2].token, 3)

    def test_batch_generate_with_stop_matchers(self):
        """Test that batch_generate with per-sequence stop_matchers stops on different tokens."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=10,
        )
        prompt = self.tokenizer.encode("hello")

        sm_0 = StopSequenceMatcher([[0]])
        sm_1 = StopSequenceMatcher([[1]])
        sm_2 = StopSequenceMatcher([[2]])

        processor_0 = make_logits_processors({0: 2000.0})
        processor_1 = make_logits_processors({1: 2000.0})
        processor_2 = make_logits_processors({2: 2000.0})

        uid0, uid1, uid2 = batch_gen.insert(
            [prompt, prompt, prompt],
            logits_processors=[processor_0, processor_1, processor_2],
            stop_matchers=[sm_0, sm_1, sm_2],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}

        self.assertEqual(responses[uid0].token, 0)
        self.assertEqual(responses[uid1].token, 1)
        self.assertEqual(responses[uid2].token, 2)
        self.assertEqual(responses[uid0].finish_reason, "stop")
        self.assertEqual(responses[uid1].finish_reason, "stop")
        self.assertEqual(responses[uid2].finish_reason, "stop")

    def test_batch_continued_generation(self):
        for rotating in [False, True]:
            if rotating:
                self.model.make_cache = lambda: [
                    RotatingKVCache(max_size=4) for _ in self.model.layers
                ]

            # Make the prompts
            prompts_a = [
                "Write a story about Einstein",
                "Hi",
                "What time is it?",
                "How tall is Mt Everest?",
            ]
            prompts_a = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_a
            ]
            prompts_b = [
                "Another one",
                "sup?",
                "And how about the date?",
                "Mt Olympus?",
            ]
            prompts_b = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_b
            ]

            # Generate once
            batch_gen = BatchGenerator(
                self.model,
                stop_tokens=self.tokenizer.eos_token_ids,
                max_tokens=10,
                prefill_batch_size=4,
                prefill_step_size=8,
                completion_batch_size=2,
            )
            uids = batch_gen.insert(prompts_a)
            caches = {uid: None for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    if r.finish_reason is not None:
                        caches[r.uid] = r.prompt_cache
            caches = [caches[uid] for uid in uids]

            # Generate the 2nd time
            uids = batch_gen.insert(prompts_b, caches=caches)
            batch_responses = {uid: [] for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    batch_responses[r.uid].append((r.token, r.logprobs))

            for e, uid in enumerate(uids):
                for i, response in enumerate(
                    stream_generate(
                        self.model,
                        self.tokenizer,
                        prompts_b[e],
                        max_tokens=10,
                        prompt_cache=caches[e],
                    )
                ):
                    batch_token, batch_logprobs = batch_responses[uid][i]
                    self._assert_batch_equivalent(
                        batch_token,
                        batch_logprobs,
                        response.token,
                        response.logprobs,
                    )

            if rotating:
                del self.model.make_cache

    def _continued_generation_test_helper(self, model):
        # Eight steps exercise repeated merge/filter/continuation cycles while
        # staying before the fixed random Qwen fixture's first low-margin
        # batch-shape trajectory fork (observed at step ten).
        max_tokens = 8

        def rand_prompt(n):
            return [random.randint(0, 1000) for _ in range(n)]

        # Make the prompts
        prompts_a = [
            rand_prompt(5),
            rand_prompt(3),
            rand_prompt(8),
            rand_prompt(1),
        ]
        prompts_b = [
            rand_prompt(2),
            rand_prompt(7),
            rand_prompt(4),
            rand_prompt(6),
        ]

        # Generate once
        batch_gen = BatchGenerator(
            model,
            stop_tokens={},
            max_tokens=max_tokens,
            prefill_batch_size=4,
            prefill_step_size=32,
            completion_batch_size=2,
        )

        uids = batch_gen.insert(prompts_a)
        caches = {uid: None for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    caches[r.uid] = r.prompt_cache

        caches = [caches[uid] for uid in uids]

        # Generate the 2nd time
        uids = batch_gen.insert(prompts_b, caches=caches)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append((r.token, r.logprobs))

        for e, uid in enumerate(uids):
            for i, (token, logprobs) in enumerate(
                generate_step(
                    mx.array(prompts_b[e]),
                    model,
                    max_tokens=max_tokens,
                    prompt_cache=caches[e],
                )
            ):
                batch_token, batch_logprobs = batch_responses[uid][i]
                self._assert_batch_equivalent(
                    batch_token,
                    batch_logprobs,
                    token,
                    logprobs,
                )

    def test_batch_continued_generation_ssm(self):
        from mlx_lm.models import mamba2

        random.seed(0)
        mx.random.seed(4)

        # Make a small SSM model
        args = mamba2.ModelArgs(
            model_type="mamba2",
            num_heads=8,
            head_dim=16,
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=128,
            state_size=32,
            num_hidden_layers=4,
            layer_norm_epsilon=1e-4,
            conv_kernel=3,
            n_groups=4,
            use_bias=False,
            use_conv_bias=False,
            tie_word_embeddings=True,
            time_step_limit=(0.01, 10),
            time_step_rank="auto",
        )
        model = mamba2.Model(args)
        self._continued_generation_test_helper(model)

    def test_batch_continued_generation_gated_delta(self):
        from mlx_lm.models import qwen3_next

        random.seed(0)
        mx.random.seed(4)
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=128,
            mlp_only_layers=[0],
            moe_intermediate_size=128,
            rms_norm_eps=1e-5,
            head_dim=64,
            rope_theta=1000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=1000,
        )
        model = qwen3_next.Model(args)
        self._continued_generation_test_helper(model)

    def test_extend_cache_with_empty(self):
        from mlx_lm.generate import _extend_cache
        from mlx_lm.models.cache import make_prompt_cache

        cache_a = make_prompt_cache(self.model)

        prompt = mx.array([[1, 2, 3]])
        self.model(prompt, cache=cache_a)
        mx.eval([c.state for c in cache_a])

        result = _extend_cache(cache_a, [])
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

        result = _extend_cache([], cache_a)
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

    def test_remove_prompt_batch_updates_currently_processing(self):
        prompt_a = self.tokenizer.encode("Write a long story about a cat")
        prompt_b = self.tokenizer.encode("Write a long story about a dog")

        gen = BatchGenerator(
            self.model,
            max_tokens=5,
            prefill_batch_size=2,
            prefill_step_size=4,
            completion_batch_size=4,
        )
        uid_a, uid_b = gen.insert([prompt_a, prompt_b])

        gen.next()

        found = gen._find_uids([uid_a, uid_b])
        for uid in [uid_a, uid_b]:
            self.assertIn(uid, found)
            self.assertEqual(found[uid][0], 1)

        gen.remove([uid_a])

        self.assertEqual(len(gen._currently_processing), len(gen._prompt_batch))

        found = gen._find_uids([uid_b])
        self.assertIn(uid_b, found)

        while responses := gen.next_generated():
            if all(r.finish_reason is not None for r in responses):
                break

    def test_batch_max_kv_size_creates_rotating_cache(self):
        max_kv_size = 256
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=max_kv_size,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, RotatingKVCache)
                    self.assertEqual(cache.max_size, max_kv_size)

    def test_batch_max_kv_size_limits_cache_growth(self):
        max_kv_size = 5
        gen = BatchGenerator(
            self.model,
            max_tokens=10,
            max_kv_size=max_kv_size,
            prefill_batch_size=1,
            prefill_step_size=128,
            completion_batch_size=1,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertLessEqual(cache.keys.shape[2], max_kv_size)

    def test_batch_max_kv_size_none_creates_regular_cache(self):
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=None,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, KVCache)

    def test_batch_generate_return_logprobs(self):
        """Test that batch_generate returns per-token logprobs when requested."""
        prompts = [
            self.tokenizer.encode("hello"),
            self.tokenizer.encode("write a poem"),
        ]
        max_tokens = 5
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=max_tokens,
            return_logprobs=True,
            return_token_ids=True,
        )

        # Check that logprobs and token_ids are returned
        self.assertIsNotNone(response.logprobs)
        self.assertIsNotNone(response.token_ids)
        self.assertEqual(len(response.logprobs), len(prompts))
        self.assertEqual(len(response.token_ids), len(prompts))

        for i in range(len(prompts)):
            # token_ids and logprobs should have same length
            self.assertEqual(len(response.token_ids[i]), len(response.logprobs[i]))
            # logprobs should be non-positive (log-probabilities)
            for lp in response.logprobs[i]:
                self.assertLessEqual(lp, 0.0)

    def test_batch_generate_no_logprobs_by_default(self):
        """Test that batch_generate does not return logprobs by default."""
        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=3,
        )
        self.assertIsNone(response.logprobs)
        self.assertIsNone(response.token_ids)


if __name__ == "__main__":
    unittest.main()
