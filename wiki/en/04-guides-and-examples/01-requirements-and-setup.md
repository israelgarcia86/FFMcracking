# 01. Requirements and Setup

Navigation: [Wiki Home](../index.md) > [04. Practical Guides](01-requirements-and-setup.md) > Requirements and Setup

================================================================================
1. SYSTEM REQUIREMENTS
================================================================================

To run FFMcracking simulations, the host system must meet the following requirements:

1. Abaqus Finite Element Suite:
   - Requires Abaqus CAE and Abaqus/Standard (version 2020 or later recommended).
   - The command `abaqus` must be accessible in your system PATH.
   - Note: Legacy versions (such as Abaqus 6.14) have known memory and degree-of-freedom limitations when using XFEM.

2. Python Environment:
   - No separate standalone Python installation is needed.
   - Simulations execute strictly via Abaqus embedded Python:
     `abaqus cae noGUI=script_name.py`
   - Depending on your Abaqus release, this interpreter is Python 2.7 (releases 2020 through 2023) or Python 3.10+ (Abaqus 2024+). FFMcracking code is structured for dual compatibility.

3. Operating System:
   - Windows 10/11 (64-bit) or officially supported Linux distributions (RHEL, Rocky Linux, SUSE, Ubuntu with certified dependencies).

================================================================================
2. SOFTWARE DEPENDENCIES
================================================================================

FFMcracking maintains a strict ZERO THIRD-PARTY DEPENDENCY policy:
- No external scientific packages required (no SciPy, Scikit-Learn, PyTorch, or Pandas).
- Entirely built on standard Python libraries (`math`, `random`, `os`, `sys`) and `numpy` (pre-bundled with Abaqus).

Repository layout:
```text
FFMcracking/
|-- src/
|   |-- FFMcracking.py         # Main solver interface
|   +-- modOptimization.py     # Bayesian optimization engine
+-- samples/
    |-- sample1_OpenHoleCrackFixed/
    |   |-- example1.py        # Fixed crack geometry calculation
    |   +-- OpenHoleLongDEF-fmesh-00.inp
    +-- sample2_OpenHoleBasicOptimLenght/
        |-- example2.py        # Crack length optimization
        +-- OpenHoleLongDEF-fmesh-00.inp
```

================================================================================
3. RUNTIME CONFIGURATION
================================================================================

To allow sample scripts to find `src/` modules, choose one of two approaches:

3.1. Method A: PYTHONPATH Configuration (Recommended)
--------------------------------------------------------------------------------
Add `src/` to your `PYTHONPATH` before launching simulations:

On Windows (PowerShell):
```powershell
$env:PYTHONPATH = "C:\path\to\FFMcracking\src;" + $env:PYTHONPATH
```

On Linux (Bash):
```bash
export PYTHONPATH="/path/to/FFMcracking/src:$PYTHONPATH"
```

3.2. Method B: Local Module Placement
--------------------------------------------------------------------------------
Alternatively, copy `FFMcracking.py` (and `modOptimization.py` for example 2) directly into the directory containing `example1.py` or `example2.py`.

================================================================================
4. PRE-FLIGHT CHECK
================================================================================

Before running simulations, verify that the `abaqus` CLI command is functional:

```powershell
abaqus whereami
```

If it reports executable paths and version information, your environment is ready.

================================================================================
Navigation: [Previous: Optimization Module](../03-modules/04-optimization-module.md) | [Wiki Home](../index.md) | [Next: Example 1 Fixed Crack](02-example-1-fixed-crack.md)
================================================================================
