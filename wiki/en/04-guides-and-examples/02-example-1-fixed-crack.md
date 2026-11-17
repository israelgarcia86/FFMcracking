# Example 1: fixed crack

The script `samples/sample1_OpenHoleCrackFixed/example1.py` computes the load factor for a straight crack defined by `(1.0, 0.0, 0.0)` and `(4.5, 0.0, 0.0)`. The candidate length is 3.5 in the model's units. The configured properties are `sigmac = 70.0` and `Gc1 = 0.4`.

## Setup and execution

Copy `src/FFMcracking.py` into the example directory or configure the module search path. Set that directory in Abaqus/CAE with **File → Set Work Directory**, then run `example1.py` using **File → Run Script**. Alternatively, from that directory run `abaqus cae noGUI=example1.py`.

The script uses the current working directory for input and `compFiles/` for output. Check that `OpenHoleLongDEF-fmesh-00.inp` is in the directory and that the model contains the `Loading` step.

## Calculation

The program extracts uncracked stresses, evaluates the stress criterion, runs the cracked XFEM model and evaluates the energy criterion. The function returns `[t_CC, t_SC, t_EC]`, while the script prints only `t_CC = max(t_SC, t_EC)`.

One Abaqus 2024 run printed approximately **3.9020**. This is a result for that configuration, not a universal value or an experimental validation. Check the model's units and inputs when interpreting it.

The uncracked result in `compFiles/` may be reused. If the `.inp` or load step changes, use a clean output directory or remove the old ODB after confirming that no running analysis is using it.

[Index](../index.md) · [Example 2](03-example-2-length-optimization.md)
