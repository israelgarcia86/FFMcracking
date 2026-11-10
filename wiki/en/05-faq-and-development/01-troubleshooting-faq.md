# 01. Troubleshooting and FAQ

Navigation: [Wiki Home](../index.md) > [05. FAQ and Development](01-troubleshooting-faq.md) > Troubleshooting and FAQ

================================================================================
1. FREQUENTLY ASKED QUESTIONS AND DIAGNOSTICS
================================================================================

1.1. Lock File Error (.lck)
--------------------------------------------------------------------------------
Symptom:
Abaqus halts reporting that the output database or job cannot be opened because a `.lck` file already exists in the output directory.

Cause:
If a previous simulation was interrupted abruptly (Ctrl+C, syntax error, or crash), Abaqus leaves a `.lck` lock file that prevents concurrent file access.

Solution:
Delete all `.lck` files from `compFiles/` before re-running:

On Windows (PowerShell):
```powershell
Get-ChildItem -Path compFiles -Filter *.lck | Remove-Item -Force
```

On Linux (Bash):
```bash
rm -f compFiles/*.lck
```

1.2. Multiplier t_SC Returns 'inf' or Negative Values
--------------------------------------------------------------------------------
Symptom:
The stress criterion returns `inf` or a negative multiplier.

Causes and fixes:
- 'inf' case: Occurs when `max(snn) <= 0`. The entire crack path is under pure compression. In pure Mode I opening, no tensile stress exists to drive the crack. Check loading directions in your `.inp` file.
- Negative case: Occurs when the candidate path starts in tension near the notch but penetrates into a compressive region. Because `evaluate_stress_criterion` divides by `min(snn)`, a negative minimum produces a negative factor. Restrict search upper bounds so the candidate crack stays within tensile zones.

1.3. Outdated Results After Modifying .inp File
--------------------------------------------------------------------------------
Symptom:
Modifying mesh, geometry, or loads in `.inp` does not alter `t_SC` or baseline results.

Cause:
Because of the uncracked model caching strategy, if `<jobName>-without.odb` exists in `outRoute`, FFMcracking skips the uncracked job and reads cached results.

Solution:
Whenever `.inp` or load step names change, clean or delete files in `compFiles/` to force a clean re-run.

1.4. Numerical Sensitivity in Betti RE Evaluation
--------------------------------------------------------------------------------
Symptom:
For extremely short crack lengths (l < 0.05 mm), RE or t_EC exhibits oscillations or unexpected signs.

Cause:
Reciprocal work RE computes differences between cracked and uncracked boundary reactions and displacements. For tiny cracks, both states are nearly identical and subtraction loses numerical precision.

Solution:
- Ensure minimum search bounds (L_min) are at least as large as one element size at the crack root.
- Set Abaqus jobs to solve with double precision (`precision=DOUBLE_PLUS_PACK`).

1.5. CAE Viewport or Session Collisions
--------------------------------------------------------------------------------
Symptom:
Errors invoking `session.Path` or `session.viewports` indicating elements already exist or are unavailable.

Cause:
Interactive session crashes may leave orphaned objects in Abaqus GUI session memory.

Solution:
Always execute via non-interactive mode (`abaqus cae noGUI=...`), which runs in a clean, isolated session environment.

================================================================================
Navigation: [Previous: Example 2](../04-guides-and-examples/03-example-2-length-optimization.md) | [Wiki Home](../index.md) | [Next: Developer Guide](02-development-and-extensions.md)
================================================================================
