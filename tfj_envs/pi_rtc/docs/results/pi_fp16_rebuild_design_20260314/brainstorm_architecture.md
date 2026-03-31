# FP16 Rebuild Brainstorm Report: Architecture / Performance

## Role

Architecture and performance analysis role. This report focuses on why the current TensorRT FP32 path
does not win on `pipeline_chunk`, what hotspots FP16 should target, and what architecture constraints
must remain stable while rebuilding.

## 1. Current Architecture Breakdown

The current PI0.5 TensorRT inference path is not a single monolithic engine. It is a three-stage pipeline:

1. `vision_encoder`
   - Export wrapper: `Pi05VisionEncoderExportWrapper`
   - Source boundary: `paligemma_with_expert.embed_image(image)`
   - Role: run the visual tower for each camera independently and emit image embeddings

2. `prefix_cache`
   - Export wrapper: `Pi05PrefixCacheExportWrapper`
   - Source boundary: build prefix embeddings from:
     - `image_embs_top`
     - `image_embs_wrist`
     - language token embeddings
   - Then call `paligemma_with_expert.forward(... inputs_embeds=[prefix_embs, None], use_cache=True)`
   - Role: prefill the prefix and materialize `past_key_values`

3. `denoise_step`
   - Export wrapper: `Pi05DenoiseStepExportWrapper`
   - Source boundary: build suffix/action embeddings from `x_t + timestep`, then call
     `paligemma_with_expert.forward(... inputs_embeds=[None, suffix_embs], past_key_values=...)`
   - Role: run one action-expert denoise iteration and emit `v_t`

Runtime orchestration is in `TrtPi05PolicyAdapter.predict_action_chunk()`:

- run vision twice
- run prefix once
- run denoise `num_inference_steps` times

This means performance depends on:

- engine latency for each stage
- repeated engine boundary overhead
- cache tensor size and movement cost
- number of denoise iterations

## 2. Benchmark Observations

From the current benchmark reports:

### FP32 / chunk benchmark

- PyTorch FP32 `pipeline_chunk`: `94.934 ms`
- TensorRT FP32 `pipeline_chunk`: `123.053 ms`

Stage breakdown:

- PyTorch FP32 `vision_encoder_pair`: `8.110 ms`
- TensorRT FP32 `vision_encoder_pair`: `12.797 ms`

- PyTorch FP32 `prefix_cache`: `25.028 ms`
- TensorRT FP32 `prefix_cache`: `63.059 ms`

- PyTorch FP32 `denoise_step`: `6.286 ms`
- TensorRT FP32 `denoise_step`: `4.650 ms`

Interpretation:

- TensorRT already helps on `denoise_step`
- TensorRT does not currently help on `vision`
- TensorRT is dramatically worse on `prefix_cache`
- `prefix_cache` dominates the whole chunk comparison

### 1000-step select_action benchmark

TensorRT wins on amortized `select_action` throughput because only a subset of control steps refresh the chunk.
That result is valid, but it does not invalidate the chunk benchmark. The two measurements answer different questions.

## 3. Primary Bottleneck Judgment

The main bottleneck is not the action expert. It is `prefix_cache`.

Why this matters:

- `denoise_step` is the part most people expect TensorRT to accelerate, and it already does
- But the current chunk path pays a very large fixed cost before denoise even starts
- With `num_inference_steps = 10`, the denoise acceleration is not enough to amortize the prefix loss

Evidence pointing to `prefix_cache` as the hotspot:

- latency delta is largest there
- engine size is abnormally large
- stage semantics are cache-heavy and output-heavy
- this stage emits a large `past_key_values` set, which is more bandwidth-sensitive than compute-dense

Working hypothesis:

- FP16 alone may help, but only if `prefix_cache` becomes materially cheaper in both compute and cache I/O
- If FP16 only improves `denoise_step`, the whole chunk may still lose

## 4. FP16 Rebuild Goal and Non-goal

### Goals

1. Rebuild all three engines in lower precision with traceable metadata
2. Preserve the current staged contract:
   - vision outputs
   - prefix cache tensor names and shapes
   - live `timestep`
3. Make `pipeline_chunk` faster than the current FP32 TRT baseline
4. Make `prefix_cache` materially faster than current FP32 TRT
5. Keep Stage 5 numerical drift within explicit, documented thresholds

### Non-goals

1. Changing the model export boundary in this iteration
2. Replacing the three-engine architecture with a monolithic engine
3. Optimizing real-robot I/O in the same change
4. Claiming global TensorRT superiority from one benchmark

## 5. Recommended Validation Metrics

FP16 success should not be judged by a single metric.

### Required performance metrics

1. `vision_encoder_single`
2. `vision_encoder_pair`
3. `prefix_cache`
4. `denoise_step`
5. `pipeline_chunk`
6. `1000-step select_action` pure inference

### Required comparison axes

1. FP32 TRT vs FP16 TRT
2. FP16 TRT vs PyTorch FP32
3. FP16 TRT vs PyTorch AMP
4. FP16 TRT vs ONNX runtime CUDA path

### Required acceptance logic

1. `denoise_step` must not regress versus TRT FP32
2. `prefix_cache` must improve enough to move total chunk latency
3. `pipeline_chunk` should beat TRT FP32 by a visible margin
4. Best case target: approach or beat current PyTorch FP32 chunk number

## 6. Suggested Success Thresholds

These are engineering targets, not hard scientific truths:

1. Minimum acceptable:
   - `pipeline_chunk` improves over current TRT FP32 by at least 10 percent
   - no Stage 5 contract break

2. Preferred:
   - `prefix_cache` improves by at least 20 percent
   - `denoise_step` improves by at least 15 percent
   - `pipeline_chunk` improves by at least 15 percent

3. Strong success:
   - `pipeline_chunk` reaches or beats current PyTorch FP32

## 7. Architecture Constraints for the Implementation

1. Do not change the semantic three-stage decomposition in the first FP16 rebuild
   - changing precision and changing architecture in the same patch will make attribution unclear

2. Preserve the exact cache naming contract
   - `past_key_values.layer_XX.key/value`
   - downstream adapter logic depends on this

3. Preserve live `timestep`
   - no fake constant folding of timestep in denoise

4. Keep benchmark methodology identical across FP32 and FP16
   - same deterministic input batch
   - same `num_inference_steps`
   - same `n_action_steps`
   - same GPU

5. Keep runtime provenance strict
   - FP16 engines must carry new metadata so runtime never mixes FP32 and FP16 artifacts accidentally

## 8. Architectural Recommendation

Proceed in two phases:

1. Phase A: pure precision rebuild
   - rebuild FP16 engines
   - no graph boundary changes
   - no runtime architecture changes
   - re-run Stage 4, Stage 5, chunk benchmark, and 1000-step benchmark

2. Phase B: only if Phase A still loses badly on chunk
   - investigate `prefix_cache` structure
   - consider reducing engine boundary overhead or cache materialization cost

This keeps the first experiment interpretable.
