# Requirements and setup

The examples require Abaqus/CAE, Abaqus/Standard, access to a license, and NumPy in the Abaqus Python environment. The execution described here was checked with Abaqus 2024; not all supported releases have been verified.

## Modules

Copy `src/FFMcracking.py` into each example directory. For example 2, also copy `src/modOptimization.py`. Alternatively, configure the Abaqus module search path to include `src`. NumPy is a dependency in addition to the Python standard library.

## Execution

Set the working directory to the example directory before running it. In Abaqus/CAE, use **File → Set Work Directory**, then **File → Run Script**. In a terminal, change to the example directory and run:

```text
abaqus cae noGUI=example1.py
```

For the second example, use `example2.py`.

The scripts obtain the input directory using `os.getcwd()` and write output to `compFiles/`. If a different directory is selected, the `.inp` file may not be found. The analyses generate files in that directory.

[Index](../index.md)
