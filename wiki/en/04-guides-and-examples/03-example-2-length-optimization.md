# Example 2: crack-length optimization

The script `samples/sample2_OpenHoleBasicOptimLenght/example2.py` searches for the lowest coupled factor among candidate crack lengths. The crack starts at `(1.0, 0.0, 0.0)` and ends at `(1.0 + L, 0.0, 0.0)`. The configured range is `L_min = 0.1` to `L_max = 5.0`, in model units.

## Setup and execution

Keep the example folder under `samples/` in the complete repository. The script imports `src.FFMcracking` and `src.modOptimization` directly. Set the example directory as the working directory and run `example2.py` in Abaqus/CAE. From a terminal in that directory, use `abaqus cae noGUI=example2.py`. Output is written to `compFiles/`.

## Optimization and interpretation

The objective returns the coupled, stress and energy factors. The algorithm evaluates five random initial lengths and ten further candidates. It reuses the uncracked ODB if it exists; each new length requires a cracked analysis. Output includes the best observed result and the evaluated points.

One Abaqus 2024 run found an observed minimum factor of **3.7411** at `L ≈ 2.0303`. Initial points are random, so the result can vary and is not guaranteed to be the global minimum. These figures describe that run, not experimental validation. See the [optimization module](../03-modules/04-optimization-module.md) for the method and its limitations.

If the model or load step changes, use a clean output directory or remove the old uncracked ODB after confirming it is not in use.

[Index](../index.md) · [Example 1](02-example-1-fixed-crack.md)
