# Applying the optional LaSA integration

The patch targets clean upstream revision
`f823dc8354e90251a0bd2f2b89fbd14eb4a9d047` and only changes
`libs/postprocess.py` plus a small NumPy compatibility fix in
`libs/metric.py` (`np.bool` to `np.bool_`). That fix restores NumPy 1.24+
compatibility without changing the strict local-maximum rule.

1. Copy `kinematic_boundary.py` from this demo to the LaSA repository root.
2. From the LaSA repository root, apply `integration.patch`:

   ```bash
   git apply /path/to/lasa-kinematic-boundary-demo/integration/integration.patch
   ```

3. Existing calls that pass only `outputs`, `boundaries`, and `masks` retain
   LaSA's original refinement path. To opt in, pass `com_speed` and
   `angular_velocity` arrays shaped `(N, T)`, plus scalar or per-sample
   reconstruction `quality` in `[0, 1]`:

   ```python
   predictions = postprocessor(
       outputs,
       boundaries,
       masks,
       com_speed=com_speed,
       angular_velocity=angular_velocity,
       quality=quality,
   )
   ```

   `masks` remains LaSA's `(N, 1, T)` boolean mask. Missing cues/QC or QC below
   0.5 use original local maxima. The patch flattens each valid boundary trace
   before peak selection and slices the `(N, T)` motion cues using the same
   valid-frame mask.

The interface patch was exercised by the optional integration test against a
temporary clean copy of the upstream files. It does not include LaSA source or
weights in this repository.

To rerun that check against a clean checkout at the verified commit, pass its
directory explicitly:

```bash
python integration/test_upstream.py --upstream-dir /path/to/LaSA
```

The script checks the Git blob IDs of the three imported upstream files,
normalizing Windows CRLF checkout to the committed LF form, applies the checked-in patch to a disposable copy, imports LaSA's actual
`PostProcessor`, and checks baseline/fallback equality, an active shifted
candidate, `(N,1,T)` masks, `(N,T)` cues, per-sample quality, and a shorter
valid sequence. It neither downloads nor modifies the supplied checkout.

