# FP16 Rebuild Brainstorm Report: Precision / Build / Code Changes

## Role

Precision and build chain analysis role. This report focuses on what code needs to change for an FP16
TensorRT rebuild, how metadata and verification should evolve, and what implementation order minimizes risk.

## 1. Current FP32 Build Chain Summary

Current Stage 4 already supports lower precision at the CLI level:

- `scripts/step4_build_engines.py`
  - `--precision {fp32, fp16, bf16}`
  - `--allow-tf32`
  - `--force-fp32-layer-types`

- `scripts/build_pi_trt_engine.py`
  - sets TensorRT builder flags based on requested precision
  - records precision in the per-engine build report
  - currently validates static-shape ONNX and writes timing cache

Observed fact:

- the verified run currently uses FP32 build settings
- metadata and benchmark docs are written around that verified FP32 run

## 2. Minimal Code-change Strategy

The minimal viable FP16 rebuild does not need a graph redesign. It needs:

1. A clean Stage 4 rebuild path for FP16
2. Stage 5 verification against the new FP16 engines
3. Benchmark scripts updated to compare FP16 and FP32 explicitly
4. Documentation updates so runtime provenance is unambiguous

This implies code changes in four areas:

### A. Build entrypoints

- `scripts/step4_build_engines.py`
- `scripts/build_pi_trt_engine.py`

### B. Verification/reporting

- `scripts/step5_verify_trt.py`

### C. Benchmarking

- `scripts/benchmark_pi_inference.py`

### D. Runtime provenance and operator UX

- `scripts/run_pi05_trt_infer_so101.py`
- documentation files

## 3. Recommended Concrete Changes

### 3.1 `step4_build_engines.py`

Recommended changes:

1. Add precision to top-level stage report in a more prominent, normalized field
2. Add per-subgraph effective precision summary to the result payload
3. Add an explicit `engine_variant` or `artifact_variant` field such as:
   - `trt_fp32`
   - `trt_fp16`
4. Encode precision into default output directories or report metadata to avoid ambiguity

Why:

- current run directories can otherwise look identical except for internal JSON fields
- runtime and benchmark tooling need a cheap way to distinguish FP16 from FP32 artifacts

### 3.2 `build_pi_trt_engine.py`

Recommended changes:

1. Keep FP16 builder flag logic as the primary path
2. Record more detailed build-time precision telemetry:
   - requested precision
   - effective TensorRT flags
   - whether fast FP16 path is available
3. If precision constraints are used later, record matched layers explicitly
4. Consider optional support for a stable default `--force-fp32-layer-types` escape hatch for FP16 debug

Why:

- first FP16 build may need selective fallback
- build report must explain exactly what precision was requested and what TensorRT actually accepted

### 3.3 `step5_verify_trt.py`

Recommended changes:

1. Add explicit reporting of engine precision variant for the verified artifacts
2. Preserve current export-fidelity comparison as the acceptance baseline
3. Consider optional relaxed thresholds for FP16 exploration, but keep default gate conservative
4. Add a separate note block for precision-related drift rather than mixing it with general runtime notes

Why:

- FP16 will likely drift more than FP32
- we need to know whether any threshold change is a deliberate policy decision

### 3.4 `benchmark_pi_inference.py`

Recommended changes:

1. Add explicit `artifact_label` or `trt_variant` field in output
2. Make it easy to run:
   - PyTorch FP32
   - PyTorch AMP
   - TRT FP32
   - TRT FP16
3. Add side-by-side comparison helper section in markdown report
4. Keep identical input and timing boundaries across variants

Why:

- benchmark output must tell the reader exactly what was compared
- FP16 rebuild is not meaningful if reports still look like generic TRT runs

### 3.5 `run_pi05_trt_infer_so101.py`

Recommended changes:

1. Surface precision/variant in startup summary
2. Refuse mixed artifacts if metadata says one precision but actual engine provenance implies another
3. Keep real-robot safety checks independent from precision choice

Why:

- operators need to know whether they are launching FP32 or FP16 artifacts
- accidental artifact mixups are very likely during repeated rebuild experiments

## 4. CLI / Metadata / Report Additions

Recommended new or clearer fields:

1. In build report:
   - `requested_precision`
   - `effective_precision_flags`
   - `engine_variant`

2. In metadata:
   - `trt_precision`
   - `trt_variant`
   - `build_settings.precision`

3. In benchmark reports:
   - `trt_variant`
   - `source_run_dir`
   - direct link to Stage 4 and Stage 5 reports used for this benchmark

4. In runtime summary:
   - `artifact_precision`
   - `artifact_variant`

## 5. Compatibility and Rollback Strategy

Recommended rollback design:

1. Do not overwrite current verified FP32 run
2. Produce a separate FP16 run directory
3. Keep benchmark scripts able to compare any two run directories
4. Keep real-robot launcher backward-compatible with FP32

Fallback path if FP16 fails:

1. Try selective FP32 layer constraints only for the failing subgraph
2. If still unstable, revert that subgraph to FP32 build for debugging only
3. Do not silently ship mixed precision artifacts as if they were pure FP16

## 6. Recommended Implementation Order

1. Build metadata foundation
   - make precision/variant reporting explicit

2. Rebuild Stage 4 with FP16 into a new run directory

3. Re-run Stage 5 verification against the new FP16 artifacts

4. Run the existing chunk benchmark and 1000-step pure inference benchmark on FP16

5. Update docs only after the results are real

6. If FP16 chunk still loses, start a second patch for deeper runtime changes

## 7. Minimal Patch Boundary

The first implementation patch should stay focused:

- yes:
  - Stage 4 precision rebuild changes
  - metadata/report improvements
  - benchmark/report updates

- no:
  - graph boundary redesign
  - multi-engine fusion
  - robot control loop redesign
  - unrelated runtime refactors

This keeps FP16 attribution clean.
