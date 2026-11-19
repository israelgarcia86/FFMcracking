# Sample 1: Calculating required load factor for a crack onset according to the coupled criterion

This sample shows how the code is able to calculate the load factor predicted by the coupled criterion of the finite fracture mechanics for a given crack geometry. The crack geometry is included as an input parameter, in this example as a segment, but it could be extended to other crack geometries.

## Usage

1. Check that Abaqus/CAE and Abaqus/Standard can access a license; connect to your license network or VPN if required.
2. Copy `src/FFMcracking.py` from the same repository revision into this example folder. The folder must also contain `example1.py` and `OpenHoleLongDEF-fmesh-00.inp`.
3. Open Abaqus/CAE and create a Standard/Explicit model database. Save any work you need before running: the script closes the active model database.
4. Select **File → Set Work Directory**, enter the absolute path of this example folder and click **OK**. The script uses this working directory to locate the input file and create `compFiles/`.
5. Select **File → Run Script**, choose `example1.py` and click **OK**. Wait for the analyses to complete. The script prints `The critical factor is: ...` in the Abaqus message area; it may also be recorded in `abaqus.rpy`.

To run from a terminal, first change to this example directory and execute `abaqus cae noGUI=example1.py`.

An Abaqus 2024 run with the supplied example returned approximately `3.9020`. This is an example result, not a universal value or an experimental validation. The script may reuse an existing uncracked ODB and removes files matching the cracked-job pattern in `compFiles/`. Use a separate example folder to preserve previous results; use fresh output after changing the input model or load step.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[APACHE 2.0](https://www.apache.org/licenses/LICENSE-2.0)
