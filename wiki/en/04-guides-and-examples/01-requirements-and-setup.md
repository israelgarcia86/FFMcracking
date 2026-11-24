# Requirements and setup

The examples require Abaqus/CAE, Abaqus/Standard, access to a license, and NumPy in the Abaqus Python environment. The execution described here was checked with Abaqus 2024; not all supported releases have been verified.

## Modules

Use the complete repository and keep each example folder under `samples/`. Both examples import `src.FFMcracking` directly. Example 2 also imports `src.modOptimization`.

## Execution

Set the working directory to the example directory before running it. In Abaqus/CAE, use **File → Set Work Directory**, then **File → Run Script**. In a terminal, change to the example directory and run:

```text
abaqus cae noGUI=example1.py
```

For the second example, use `example2.py`.

The scripts obtain the input directory using `os.getcwd()` and write output to `compFiles/`. If a different directory is selected, the `.inp` file may not be found. The analyses generate files in that directory.

[Index](../index.md)
