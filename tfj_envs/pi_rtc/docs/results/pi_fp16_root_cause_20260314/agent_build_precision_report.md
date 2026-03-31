# PI FP16 Stage 4 Build Precision Root-Cause Report

Date: 2026-03-14

## Scope

This report answers one narrow question:

Why can the FP16 Stage 4 TensorRT build pass, while Stage 5 correctness verification fails?

The analysis is based only on the local repository and local run artifacts. The primary evidence set is:

- `scripts/build_pi_trt_engine.py`
- `scripts/step4_build_engines.py`
- `docs/results/pi_model_fp16_20260314_172759/stage2_export_onnx.json`
- `docs/results/pi_model_fp16_20260314_172759/stage3_verify_onnx.json`
- `docs/results/pi_model_fp16_20260314_172759/stage4_build_engines.json`
- `docs/results/pi_model_fp16_20260314_172759/pi_trt_metadata.json`
- `docs/results/pi_model_fp16_20260314_172759/artifacts/engines/*_build_report.json`
- `docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.json`
- `docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.md`

Supplementary local comparison evidence:

- `docs/results/pi05_onnx_fix_20260311_230500/stage4_build_engines_bf16_mmfp32.json`
- `docs/results/pi05_onnx_fix_20260311_230500/engines_bf16_mmfp32/*_build_report.json`
- `docs/results/pi05_onnx_fix_20260311_230500/stage5_verify_trt_fp32.md`

## Executive Conclusion

The current evidence points much more strongly to a TensorRT low-precision kernel / fusion / accumulation problem than to an ONNX export-boundary problem.

Stage 4 pass means:

- ONNX parsed successfully in TensorRT.
- Input shapes were static.
- The requested builder flags were set.
- The engine serialized successfully.

Stage 4 pass does **not** mean:

- every numerically sensitive layer executed in FP32 internally;
- all important low-precision-sensitive kernels were covered by the forced-FP32 policy;
- Stage 5 numeric fidelity was validated.

For this FP16 run, Stage 2 and Stage 3 already show the export boundary is healthy, while Stage 5 shows large ONNX-vs-TRT drift, especially in `prefix_cache`. That pattern is much more consistent with TensorRT precision behavior after build than with a broken exporter boundary.

## What Stage 4 Actually Validates

### 1. Static shape validation is structural only

In `scripts/build_pi_trt_engine.py`, `_validate_static_shapes()` only checks that parsed network **input** dimensions are not negative:

- `build_pi_trt_engine.py:49-61`

If any input tensor shape contains a dynamic dimension, Stage 4 fails. If not, Stage 4 proceeds.

This check says nothing about:

- numerical correctness;
- per-layer effective compute precision;
- runtime drift relative to ONNX or Torch.

### 2. Precision constraints are applied only by exact TensorRT layer type match

In `scripts/build_pi_trt_engine.py`, `_apply_precision_constraints()` resolves each user-provided name with `getattr(trt.LayerType, name.upper())`, then iterates the parsed TensorRT network and applies:

- `layer.precision = trt.float32`
- `layer.set_output_type(output_index, trt.float32)`

Relevant code:

- `build_pi_trt_engine.py:162-166`
- `build_pi_trt_engine.py:169-203`

This means the constraint policy only touches layers whose parsed TensorRT `layer.type` is exactly one of the named enums.

### 3. Stage 4 only records requested flags and visible evidence

For FP16, the builder enables `trt.BuilderFlag.FP16`:

- `build_pi_trt_engine.py:270-273`

If any constrained layers were matched, the config enables:

- `OBEY_PRECISION_CONSTRAINTS` if present, else
- `PREFER_PRECISION_CONSTRAINTS`

Relevant code:

- `build_pi_trt_engine.py:282-287`

The build report then records:

- requested precision;
- builder flag state;
- forced layer-type list;
- matched layer count;
- engine input/output dtypes.

But the report itself explicitly says it does **not** prove effective per-layer execution precision:

- `build_pi_trt_engine.py:341-350`

That disclaimer is central to this investigation. Stage 4 pass is intentionally not a numerical correctness proof.

### 4. Stage 4 metadata is a passthrough, not deeper introspection

`scripts/step4_build_engines.py` forwards the same `force_fp32_layer_types` into every subgraph build:

- `step4_build_engines.py:274-284`

It records the build settings:

- `step4_build_engines.py:321-329`

and stores `trt_effective_precision_evidence` into metadata:

- `step4_build_engines.py:352-358`

This metadata is still only the same requested-flag / matched-layer / IO-dtype evidence from Stage 4, not proof of internal accumulation precision.

## What `force_fp32_layer_types = REDUCE ELEMENTWISE UNARY` Actually Constrains

For the FP16 run `docs/results/pi_model_fp16_20260314_172759/`, Stage 4 used:

- precision: `fp16`
- `allow_tf32 = false`
- `force_fp32_layer_types = ["REDUCE", "ELEMENTWISE", "UNARY"]`

Recorded in:

- `stage4_build_engines.json`
- `pi_trt_metadata.json`

Matched layer counts in the generated build reports:

| Subgraph | Matched count |
| --- | ---: |
| `vision_encoder` | 475 |
| `prefix_cache` | 2057 |
| `denoise_step` | 1788 |

These counts look large, but they are not coverage of the numerically dominant kernels. They are coverage of whatever TensorRT parsed as `REDUCE`, `ELEMENTWISE`, or `UNARY`.

Examples from the actual matched-layer evidence:

### `vision_encoder`

First matched layers are:

- `ONNXTRT_ShapeElementWise`
- `ONNXTRT_ShapeElementWise_3`
- `ONNXTRT_ShapeElementWise_6`
- ...

This shows that a noticeable portion of the match count is spent on shape-related or simple elementwise layers, not on the main attention / MLP matmuls.

### `prefix_cache`

First matched layers are:

- `/Mul`
- `/Mul_1`
- `/Equal`
- `ONNXTRT_ShapeElementWise`
- ...

Again, many matches are on light elementwise / masking / shape-style operations.

### `denoise_step`

First matched layers include:

- `node_Mul_102`
- `node_Sin_103`
- `node_Cos_104`
- `node_Add_109`
- `node_ReduceSum_270`

These are real math ops, but they still do not cover the main attention and projection matmuls.

## What the Current Policy Does **Not** Constrain

The current FP16 policy does **not** constrain any TensorRT layer whose type is not exactly one of:

- `REDUCE`
- `ELEMENTWISE`
- `UNARY`

That means it does not directly constrain:

- `MATRIX_MULTIPLY`
- `SOFTMAX`
- `CUMULATIVE`
- `CAST`
- `SHUFFLE`
- plugin layers
- fused kernels whose important accumulation happens inside other layer types
- internal accumulation details inside TensorRT-selected kernels

This is not hypothetical. The local repo already contains proof that the builder recognizes more relevant layer types.

In the historical local run:

- `docs/results/pi05_onnx_fix_20260311_230500/stage4_build_engines_bf16_mmfp32.json`

the build settings used:

- `force_fp32_layer_types = ["MATRIX_MULTIPLY", "SOFTMAX", "REDUCE", "CUMULATIVE"]`

and the corresponding build reports show explicit matches such as:

- `/layers.0/self_attn/q_proj/MatMul`
- `/layers.0/self_attn/MatMul`
- `/layers.0/self_attn/Softmax`
- `/CumSum`

So the implementation can target these layer types. The current FP16 run simply did not request them.

## Coverage Gap Against the Actual ONNX Graphs

Operator counts from the current FP16 ONNX artifacts:

### `pi_shared_vision_encoder.onnx`

- `MatMul`: 217
- `Softmax`: 27
- `LayerNormalization`: 55
- total nodes: 1314

### `pi_shared_prefix_cache.onnx`

- `MatMul`: 156
- `Softmax`: 17
- `ReduceMean`: 35
- `Sqrt`: 35
- `Div`: 35
- total nodes: 2315

### `pi05_denoise_step.onnx`

- `MatMul`: 165
- `Gemm`: 39
- `Softmax`: 18
- `ReduceMean`: 37
- total nodes: 1487

This matters because the current FP16 forced-FP32 policy mainly protects elementwise / unary / reduce nodes, while the graphs still contain a large number of unconstrained attention and projection matmuls plus softmax-heavy attention blocks.

That mismatch is exactly what I would expect if Stage 4 still passes but Stage 5 later shows large low-precision drift.

## FP16 Run Evidence: Why Stage 5 Looks Like TRT Precision Drift

## 1. Stage 2 and Stage 3 already validate the export boundary

From the current FP16 run:

- `stage2_export_onnx.json`: `overall_status = pass`
- `stage2_acceptance.immediate_export_fidelity_compare = pass`
- `stage3_verify_onnx.json`: `stage3_acceptance.status = pass`
- `stage3_verify_onnx.json`: `local_export_fidelity_compare = pass`
- `stage3_verify_onnx.json`: `chained_export_fidelity_compare = pass`
- `stage3_verify_onnx.json`: `denoise_timestep_live_input = pass`

This significantly lowers the probability that the root cause is an ONNX export-boundary bug.

In particular, the denoise timestep contract was explicitly checked and passed, so this does not look like a stale export boundary or a dropped timestep input problem.

## 2. Stage 4 pass is clean, but only as a build gate

From the current FP16 run:

- `stage4_build_engines.json`: `overall_status = pass`
- `pi_trt_metadata.json`: `stage_status.stage4_build_engines = pass`
- all three engine build reports have `status = pass`

The recorded builder evidence is:

- `FP16 = true`
- `TF32 = false`
- `OBEY_PRECISION_CONSTRAINTS = true`

That proves the requested build policy was applied. It does **not** prove the policy was sufficient.

## 3. Stage 5 primary compare is against export-fidelity ONNX, not a noisy runtime baseline

The current Stage 5 report uses:

- `primary_onnx_compare_profile = export_fidelity`

and the ONNX sessions are intentionally conservative:

- `vision_encoder`: CPU EP, `graph_optimization_level = all`
- `prefix_cache`: CPU EP, `graph_optimization_level = disable`
- `denoise_step`: CPU EP selected first, `graph_optimization_level = disable`

So the ONNX side is intentionally trying to stay close to the export boundary rather than optimizing aggressively.

## 4. Torch-vs-ONNX is tight, but ONNX-vs-TRT is not

The strongest evidence is that Torch and ONNX remain close, while TRT diverges from both.

### `vision_encoder`

`torch_vs_onnx`:

- max abs diff: `0.0004959`
- mean abs diff: `3.75e-06`
- min cosine: `0.99999994`

`onnx_vs_trt`:

- max abs diff: `0.9992447`
- mean abs diff: `0.00883515`
- min cosine: `0.99998826`

### `prefix_cache`

`torch_vs_onnx`:

- max abs diff: `0.0004063`
- mean abs diff: `7.47e-06`
- min cosine: `0.99999976`

`onnx_vs_trt`:

- max abs diff: `12.2229338`
- mean abs diff: `0.34146136`
- min cosine: `0.53000045`

### `denoise_step`

`torch_vs_onnx`:

- max abs diff: `1.55e-06`
- mean abs diff: `1.13e-07`
- min cosine: `1.0`

`onnx_vs_trt`:

- max abs diff: `0.02522576`
- mean abs diff: `0.000821512`
- min cosine: `0.99999177`

### `pipeline`

`torch_vs_onnx`:

- max abs diff: `7.39e-06`
- mean abs diff: `3.51e-07`
- min cosine: `1.0`

`onnx_vs_trt`:

- max abs diff: `0.0631877`
- mean abs diff: `0.00202293`
- min cosine: `0.99992281`

This pattern is not what I would expect from a broken export boundary. If the exporter were wrong, I would expect Stage 2 or Stage 3 to already show the defect, or Torch-vs-ONNX to degrade in the same locations.

Instead, the drift appears when TensorRT executes the built FP16 engines.

## 5. `prefix_cache` failure is broad, not a single-output glitch

For `prefix_cache`:

- total outputs compared: `37`
- failed outputs: `36`

Worst ONNX-vs-TRT outputs include:

| Output | max abs diff | mean abs diff | cosine |
| --- | ---: | ---: | ---: |
| `past_key_values.layer_01.value` | 12.2229 | 0.2160 | 0.5300 |
| `past_key_values.layer_02.value` | 9.5802 | 0.1671 | 0.6327 |
| `past_key_values.layer_03.value` | 5.6625 | 0.1038 | 0.7366 |
| `past_key_values.layer_04.value` | 5.3595 | 0.1113 | 0.7451 |
| `past_key_values.layer_01.key` | 11.7886 | 0.1445 | 0.8870 |

This is broad corruption across the KV cache, not a one-off boundary mismatch on a single output tensor.

That shape of failure is much more consistent with attention-path low-precision behavior than with exporter bookkeeping.

## Why the Evidence Favors TensorRT Kernel / Accumulation Issues

My judgment is:

The local evidence favors "TensorRT low-precision kernel selection / fusion / accumulation behavior is insufficiently constrained by the current Stage 4 build policy" over "the ONNX export boundary is wrong."

The reason chain is:

1. Stage 2 and Stage 3 show the export boundary is healthy.
2. The only major change between that healthy ONNX boundary and the failing Stage 5 run is the TensorRT build/execution path.
3. Stage 4 explicitly says it only proves requested precision flags, matched layer types, and engine IO dtypes.
4. The current forced-FP32 policy ignores the most likely attention-sensitive layer types: `MATRIX_MULTIPLY`, `SOFTMAX`, and `CUMULATIVE`.
5. The actual ONNX graphs contain many `MatMul` and `Softmax` nodes.
6. The worst failure is in `prefix_cache`, exactly where attention KV generation dominates.
7. The failure signature is ONNX-vs-TRT drift, not Torch-vs-ONNX drift.

That does **not** prove the exact offending TensorRT kernel. But it strongly narrows the class of causes to low-precision execution behavior inside TensorRT rather than exporter boundary logic.

## Additional Local Cross-Check: FP32 Stage 5

A useful local control exists in:

- `docs/results/pi05_onnx_fix_20260311_230500/stage5_verify_trt_fp32.md`

In that run, the ONNX-vs-TRT numbers are very tight:

### `vision_encoder`

- max abs diff: `1.1444e-05`
- mean abs diff: `7.34e-08`
- min cosine: `0.99999982`

### `prefix_cache`

- max abs diff: `0.0001335`
- mean abs diff: `1.00e-05`
- min cosine: `0.99999982`

### `denoise_step`

- max abs diff: `1.6689e-06`
- mean abs diff: `3.68e-07`
- min cosine: `1.0`

This is not a perfect apples-to-apples proof, but it is strong local supporting evidence that the general export path can match well once the precision regime changes. That again points away from exporter boundary breakage and toward low-precision TensorRT behavior.

## Bottom Line

### What Stage 4 constrained

It constrained only TensorRT layers parsed as:

- `REDUCE`
- `ELEMENTWISE`
- `UNARY`

and forced those matched layers plus their outputs to FP32, with `OBEY_PRECISION_CONSTRAINTS` enabled.

### What Stage 4 did **not** constrain

It did not directly constrain:

- `MATRIX_MULTIPLY`
- `SOFTMAX`
- `CUMULATIVE`
- many attention-path kernels
- internal accumulation precision of unconstrained or fused kernels

### Most likely interpretation

The current FP16 Stage 4 build can pass because the engine is structurally valid and serializable, while Stage 5 still fails because the selected FP32 overrides do not cover the numerically dominant low-precision-sensitive kernels. The observed failure pattern is much more consistent with TensorRT kernel / accumulation / fusion behavior than with an ONNX exporter boundary defect.

## Suggested Next Build-Side Checks

If the goal is to confirm this hypothesis quickly, the most informative next checks would be:

1. Rebuild FP16 with `MATRIX_MULTIPLY` and `SOFTMAX` added to `force_fp32_layer_types`, then rerun Stage 5.
2. Consider adding `CUMULATIVE` too, since the local repo already used it in a prior experiment.
3. Compare per-subgraph impact, especially `prefix_cache`, because that is where the current failure is most diagnostic.
4. If needed, inspect layer-wise TensorRT profiling or engine inspection artifacts to determine whether the attention path is still using low-precision kernels despite the current overrides.

That would directly test the current strongest hypothesis coming out of the build evidence.
