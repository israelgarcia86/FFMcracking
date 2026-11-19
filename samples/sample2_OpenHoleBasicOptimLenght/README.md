# Sample 2: Calculating optimal crack length and critical factor for a crack onset according to the coupled criterion

This sample shows how the code is able to calculate the critical load factor by minimizing the required load factor, using as an argument only the crack length at onset. The code use a ver basic Optimization Algorithm without being optimized for this particular application. Some modifications that can be employed could be the separation of the surrogate model differently for the stress criterion and for the energy criterion because are smooth functions, contrarily to the max of the two that is not smooth. 

## Usage

1. Keep this example folder under `samples/` in the complete repository. The script imports `src.FFMcracking` and `src.modOptimization` directly.
2. Set this folder as the working directory in **File → Set Work Directory** and select `example2.py` in **File → Run Script**. To run from a terminal, change to this folder and execute `abaqus cae noGUI=example2.py`.
3. Look for the critical factor and corresponding crack length in the Abaqus message area or `abaqus.rpy`. The analysis files are written to `compFiles/`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[APACHE 2.0](https://www.apache.org/licenses/LICENSE-2.0)