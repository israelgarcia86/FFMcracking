# Example 1: fixed crack

The script `samples/sample1_OpenHoleCrackFixed/example1.py` computes the load factor for a straight crack defined by `(1.0, 0.0, 0.0)` and `(4.5, 0.0, 0.0)`. The candidate length is 3.5 in the model's units. The configured properties are `sigmac = 70.0` and `Gc1 = 0.4`.

## Setup and execution

1. Check that Abaqus/CAE and Abaqus/Standard can access a license. Connect to your license network or VPN if required.
2. Keep `samples/sample1_OpenHoleCrackFixed/` in the complete repository. The folder must contain `example1.py` and `OpenHoleLongDEF-fmesh-00.inp`. The script imports `src.FFMcracking` directly.
3. Open Abaqus/CAE and create a Standard/Explicit model database. Save any work you need first: the script closes the active model database.
4. In **File → Set Work Directory**, enter the absolute path of the example folder on your computer and click **OK**. Selecting the script in the next step does not by itself change the working directory.
5. In **File → Run Script**, select `example1.py` and click **OK**. Wait for the uncracked and cracked analyses to complete. The script prints `The critical factor is: ...` in the Abaqus message area; the session replay file `abaqus.rpy` may also contain the output.

Alternatively, open a terminal in the example directory and run `abaqus cae noGUI=example1.py`.

The script uses the current working directory for input and `compFiles/` for output. Check that `OpenHoleLongDEF-fmesh-00.inp` is in the directory and that the model contains the `Loading` step.

## Calculation

The program extracts uncracked stresses, evaluates the stress criterion, runs the cracked XFEM model and evaluates the energy criterion. The function returns `[t_CC, t_SC, t_EC]`, while the script prints only `t_CC = max(t_SC, t_EC)`.

One Abaqus 2024 run printed approximately **3.9020**. This is a result for that configuration, not a universal value or an experimental validation. Check the model's units and inputs when interpreting it.

The uncracked result in `compFiles/` may be reused. The script also removes prior files matching the cracked-job pattern in that folder. Use a separate copy of the example folder under `samples/` to preserve earlier results; use fresh output if the `.inp` or load step changes.

## Visualizing the result

The script exports `crack-s22.png` and `crack-opening-s22.png` to `compFiles/visualization/`. The first shows S22 on the undeformed model at the final frame of `Loading`. The second shows the same stress field on the deformed model, with a 150× display scale to make the XFEM crack opening visible. The field corresponds to the reference load in the analysis. The scale changes the displayed shape only.

Running the script from Abaqus/CAE leaves the second view on screen. A command-line run also exports the images. To inspect the result later, open `compFiles/visualization/crack.odb` in Abaqus/CAE.

[Index](../index.md) · [Example 2](03-example-2-length-optimization.md)
