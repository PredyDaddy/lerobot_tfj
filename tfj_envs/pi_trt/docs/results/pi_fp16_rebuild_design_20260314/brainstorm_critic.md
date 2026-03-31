# FP16 Rebuild Brainstorm Report: Critic / Risk Review

## Role

Critical reviewer role. This report assumes the job is likely to fail unless the methodology and
acceptance criteria are kept strict.

## 1. Highest-risk Failure Modes

### 1.1 Prefix cache looks faster in isolation but chunk still loses

This is the most likely disappointment path.

Why:

- current chunk regression is dominated by `prefix_cache`
- if FP16 only improves `denoise_step`, headline latency will barely move

### 1.2 FP16 passes one benchmark but silently drifts numerically

Possible symptoms:

- Stage 5 still "looks okay" on coarse summaries
- but action quality degrades on long denoise loops or real tasks

Risk concentration:

- `prefix_cache` tensors
- `past_key_values`
- repeated denoise accumulation over multiple steps

### 1.3 Mixed-artifact confusion

Very likely operational mistake:

- benchmark report says FP16
- runtime accidentally points at FP32 engines
- or some engines come from one run and others from another

This is especially dangerous because the current project already has multiple run directories.

### 1.4 FP16 result gets overclaimed

Most tempting but wrong narrative:

- "TRT FP16 rebuilt successfully, therefore deployment is solved"

Wrong because:

- robot loop latency is still unmeasured here
- FP16 chunk may improve without beating PyTorch
- one run on one GPU does not generalize

## 2. Illusions That Must Be Rejected

1. "Denoise got faster, so the full pipeline is fixed"
   - false if prefix remains dominant

2. "Stage 5 passed, so FP16 is safe for deployment"
   - false unless precision-specific drift is reviewed explicitly

3. "1000-step select_action wins, so chunk latency no longer matters"
   - false because chunk generation is still the real refresh cost

4. "FP16 beats FP32 TRT, so TensorRT is now unequivocally faster than Torch"
   - false unless same measurement boundary and same precision story are stated clearly

## 3. Mandatory New Gates

The existing gates are necessary but not sufficient for FP16.

### Required acceptance additions

1. Precision provenance gate
   - every engine and benchmark must explicitly identify FP16 vs FP32

2. Cross-run coherence gate
   - benchmark should fail fast if run directory mixes artifacts from different rebuilds

3. Precision-specific Stage 5 drift summary
   - not just pass/fail
   - explicitly highlight whether FP16 increased error margins versus FP32 TRT

4. Chunk regression gate
   - FP16 should not be called successful if `pipeline_chunk` does not improve materially over TRT FP32

## 4. Minimum Benchmark Matrix

Do not accept a one-line "FP16 is faster" claim. At minimum, measure:

1. TRT FP32 vs TRT FP16
2. PyTorch FP32 vs PyTorch AMP
3. `pipeline_chunk`
4. `1000-step select_action`
5. stage breakdown:
   - `vision_encoder_pair`
   - `prefix_cache`
   - `denoise_step`

If any of these are missing, the conclusion is weak.

## 5. Documentation Constraints That Must Be Written Upfront

The docs must explicitly state:

1. FP16 results are for the current static-shape, batch=1 engines only
2. Offline benchmark does not equal robot loop latency
3. `pipeline_chunk` and `select_action` answer different questions
4. FP16 success is primarily about `prefix_cache` if chunk latency is the target
5. Real-robot rollout should still require verified-pass metadata and a dedicated preflight

## 6. Anti-patterns to Avoid

These are the most dangerous shortcuts:

1. Overwriting the current verified FP32 run directory
2. Relaxing Stage 5 thresholds without documenting why
3. Mixing FP16 and FP32 artifacts in one run directory
4. Claiming success from denoise-only wins
5. Skipping the 1000-step pure inference comparison after rebuilding

## 7. Critical Recommendation

Treat the first FP16 rebuild as an experiment, not a deployment candidate.

Required sequence:

1. separate run directory
2. explicit precision metadata
3. Stage 5 verification
4. chunk benchmark
5. 1000-step pure inference benchmark
6. only then decide whether the FP16 line deserves runtime adoption
