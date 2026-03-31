# PI FP16 `prefix_cache` Root Cause Report

## Scope

This note is based only on local repository artifacts from this run:

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage3_verify_onnx.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage4_build_engines.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/artifacts/engines/prefix_cache_build_report.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/build_pi_trt_engine.py`

## Executive Conclusion

Current FP16 deployment is blocked primarily by `prefix_cache`, not because ONNX export is wrong, but because the TensorRT FP16 engine introduces large, systemic corruption into the generated KV cache tensors.

The strongest evidence is:

- Stage 3 export-fidelity ONNX verification for `prefix_cache` is clean.
- Stage 5 `torch_vs_onnx` for `prefix_cache` is also clean.
- Stage 5 `torch_vs_trt` and `onnx_vs_trt` for `prefix_cache` are both catastrophically bad and almost identical.
- `prefix_pad_masks` is exact, while essentially every KV tensor is badly wrong.

That pattern localizes the problem to the TensorRT FP16 execution path inside the `prefix_cache` transformer/KV-generation subgraph.

## 1. Error Distribution Across `prefix_cache` Outputs

### 1.1 Stage-level summary

From `stage5_verify_trt.json`:

- `prefix_cache.status = fail`
- `torch_vs_onnx` summary:
  - `max_abs_diff = 0.0004062652587890625`
  - `mean_abs_diff = 7.473509867850225e-06`
  - `min_cosine_similarity = 0.9999997615814209`
- `torch_vs_trt` summary:
  - `max_abs_diff = 12.222936630249023`
  - `mean_abs_diff = 0.34146174788475037`
  - `min_cosine_similarity = 0.5300003886222839`
- `onnx_vs_trt` summary:
  - `max_abs_diff = 12.222933769226074`
  - `mean_abs_diff = 0.3414613604545593`
  - `min_cosine_similarity = 0.5300004482269287`

Interpretation:

- Torch and ONNX agree very closely on `prefix_cache`.
- TRT diverges heavily from both Torch and ONNX.
- The `torch_vs_trt` and `onnx_vs_trt` numbers are nearly identical, which means the error is injected after ONNX export, inside TRT execution.

### 1.2 Output-level pass/fail inventory

`prefix_cache` has 37 outputs total:

- 1 mask output: `prefix_pad_masks`
- 36 KV outputs: `past_key_values.layer_00..17.{key,value}`

Per-output Stage 5 result:

- `prefix_pad_masks` is exact and passes:
  - `max_abs_diff = 0`
  - `mean_abs_diff = 0`
  - `cosine_similarity = 1.0`
- For both `torch_vs_trt` and `onnx_vs_trt`, only `prefix_pad_masks` passes.
- All 36 KV outputs fail.

This is important: the basic discrete mask contract is intact, but the floating-point cache tensors are not.

### 1.3 Worst outputs

Worst `torch_vs_trt` outputs by `max_abs_diff`:

1. `past_key_values.layer_01.value`
   - `max_abs_diff = 12.222936630249023`
   - `mean_abs_diff = 0.2160341888666153`
   - `cosine_similarity = 0.5300003886222839`
2. `past_key_values.layer_14.value`
   - `max_abs_diff = 11.879782676696777`
   - `mean_abs_diff = 0.31644847989082336`
   - `cosine_similarity = 0.9613490700721741`
3. `past_key_values.layer_01.key`
   - `max_abs_diff = 11.788568496704102`
   - `mean_abs_diff = 0.1445278376340866`
   - `cosine_similarity = 0.8869665861129761`
4. `past_key_values.layer_17.value`
   - `max_abs_diff = 11.686826705932617`
   - `mean_abs_diff = 0.31427156925201416`
   - `cosine_similarity = 0.9639129638671875`
5. `past_key_values.layer_16.value`
   - `max_abs_diff = 11.288311004638672`
   - `mean_abs_diff = 0.3215344548225403`
   - `cosine_similarity = 0.9127788543701172`
6. `past_key_values.layer_15.value`
   - `max_abs_diff = 11.168536186218262`
   - `mean_abs_diff = 0.34146174788475037`
   - `cosine_similarity = 0.9482133388519287`

Worst outputs by cosine similarity:

1. `past_key_values.layer_01.value`: `0.5300003886222839`
2. `past_key_values.layer_02.value`: `0.6327405571937561`
3. `past_key_values.layer_03.value`: `0.7365891337394714`
4. `past_key_values.layer_04.value`: `0.7451077103614807`
5. `past_key_values.layer_05.value`: `0.7608882784843445`
6. `past_key_values.layer_06.value`: `0.7869166731834412`

Best outputs by `max_abs_diff` are still not deployment-safe:

- `past_key_values.layer_00.value`
  - `max_abs_diff = 0.04315757751464844`
  - `mean_abs_diff = 0.0016328017227351665`
  - `cosine_similarity = 0.9999953508377075`
- `past_key_values.layer_00.key`
  - `max_abs_diff = 1.409848928451538`
  - `mean_abs_diff = 0.006557574961334467`
  - `cosine_similarity = 0.9996813535690308`

Even the best KV outputs are far above the Stage 5 thresholds:

- `max_abs_diff <= 0.001`
- `mean_abs_diff <= 0.0001`
- `cosine_similarity >= 0.999`

## 2. Is the Failure Localized or Widespread?

It is clearly widespread, not concentrated in one or two layers.

### 2.1 Layer coverage

Aggregating the 36 KV outputs into 18 transformer layers:

- 18/18 layers show material drift.
- Median per-layer `max_abs_diff` for `torch_vs_trt` is about `6.44`.
- Minimum per-layer `max_abs_diff` is still about `1.41`.
- Maximum per-layer `max_abs_diff` is about `12.22`.

Top layers by per-layer `max_abs_diff`:

- Layer 1: `12.22`
- Layer 14: `11.88`
- Layer 17: `11.69`
- Layer 16: `11.29`
- Layer 15: `11.17`
- Layer 2: `9.99`

This is not “only a few late layers broke”. Early layers and late layers both drift badly.

### 2.2 Global spread statistics

Across the 36 KV outputs in `torch_vs_trt`:

- 35 outputs have `max_abs_diff > 1`
- 27 outputs have `max_abs_diff > 5`
- 31 outputs have `mean_abs_diff > 0.1`
- 12 outputs have `cosine_similarity < 0.9`
- 25 outputs have `cosine_similarity < 0.95`

That is a full-subgraph failure pattern.

### 2.3 Key vs value

Both `key` and `value` tensors are bad.

`torch_vs_trt` median metrics:

- `key`
  - median `max_abs_diff = 6.209948301315308`
  - median `mean_abs_diff = 0.1095653623342514`
  - median `cosine_similarity = 0.9472168385982513`
- `value`
  - median `max_abs_diff = 5.558603763580322`
  - median `mean_abs_diff = 0.11797266826033592`
  - median `cosine_similarity = 0.8931750655174255`

So the corruption is not isolated to only `key` or only `value`, though `value` tensors are generally worse in cosine similarity.

## 3. Why `prefix_cache` Is the Current FP16 Main Blocker

Within the same Stage 5 report:

- `vision_encoder` TRT drift is bad, but much smaller:
  - `torch_vs_trt.max_abs_diff = 0.9991912841796875`
  - `mean_abs_diff = 0.008834916166961193`
  - `min_cosine_similarity = 0.9999884963035583`
- `denoise_step` TRT drift is smaller still:
  - `torch_vs_trt.max_abs_diff = 0.025226354598999023`
  - `mean_abs_diff = 0.0008215145207941532`
  - `min_cosine_similarity = 0.999991774559021`
- `prefix_cache` is much worse:
  - `torch_vs_trt.max_abs_diff = 12.222936630249023`
  - `mean_abs_diff = 0.34146174788475037`
  - `min_cosine_similarity = 0.5300003886222839`

`pipeline` final output drift is only `max_abs_diff = 0.0631941556930542`, which is much smaller than the raw `prefix_cache` corruption. That means downstream denoise partially absorbs or smooths some cache error, but it does not make the cache trustworthy.

For deployment, `prefix_cache` is still the main blocker because it emits long-lived KV state used by the denoise stage. If the cache tensors themselves are this distorted, the runtime is fundamentally unsafe even if a later scalar/action output sometimes looks less dramatic.

## 4. Most Likely Engineering Root Cause

### 4.1 What is ruled out

The current local artifacts strongly rule out several hypotheses:

- Not an ONNX export bug:
  - Stage 3 export-fidelity verification passes for `prefix_cache`.
  - Stage 3 local `prefix_cache` summary:
    - `max_abs_diff = 0.0004062652587890625`
    - `mean_abs_diff = 7.473509867850225e-06`
    - `min_cosine_similarity = 0.9999997615814209`
- Not an input binding / output name mismatch:
  - `prefix_pad_masks` is exact.
  - Engine input/output names and dtypes line up with the contract.
- Not “Torch baseline picked the wrong mode”:
  - Stage 5 explicitly uses `export_reference_torch`.
  - `torch_vs_onnx` is clean for `prefix_cache`.

### 4.2 What the build report says

From the FP16 `prefix_cache` build report:

- `requested_precision = fp16`
- TensorRT builder flags:
  - `FP16 = True`
  - `BF16 = False`
  - `TF32 = False`
  - `OBEY_PRECISION_CONSTRAINTS = True`
- Only these layer types are forced to FP32:
  - `REDUCE`
  - `ELEMENTWISE`
  - `UNARY`
- The report explicitly notes:
  - visible I/O dtypes do not guarantee per-layer effective execution precision

This means the engine is still free to run the heavy transformer compute path, especially GEMM / QKV projection / attention-style matrix operations, in FP16 tactics internally.

### 4.3 Most likely root cause

The most likely engineering root cause is:

`prefix_cache` contains the heaviest transformer prefix-processing path in the whole PI pipeline, and the current FP16 TensorRT build only protects simple layer classes (`REDUCE`, `ELEMENTWISE`, `UNARY`) in FP32. The numerically sensitive core attention / projection / cache-generation path remains effectively FP16 or fused into FP16-dominant tactics, causing systematic KV corruption across almost every layer.

Why this fits the evidence:

- The failure appears only in TRT, not in Torch or ONNX.
- It affects nearly all KV outputs, not just one tensor.
- The discrete mask output remains exact, so the graph wiring is intact.
- The drift magnitude and cosine collapse are characteristic of internal numeric instability, not of a single missing cast or swapped tensor.
- `prefix_cache` is exactly the subgraph where long-prefix transformer activations are turned into multi-layer KV caches, so precision loss there naturally propagates to every layer’s cache outputs.

In short:

The current FP16 engine is numerically unsafe for the `prefix_cache` transformer core, most likely because the build constrains only auxiliary ops to FP32 while leaving the real cache-producing matmul/attention path in unstable FP16 TRT tactics.

## 5. Practical Takeaway

If the goal is “current FP16 can deploy”, the answer from these local artifacts is no, and `prefix_cache` is the clearest blocker.

The next engineering direction should not be more ONNX export debugging. The data says export is already fine. The next direction should be TRT precision policy for `prefix_cache`, for example:

- isolate which internal matmul/attention paths need FP32,
- introduce stronger precision constraints than only `REDUCE/ELEMENTWISE/UNARY`,
- or treat `prefix_cache` as a mixed-precision / non-FP16-safe subgraph.
