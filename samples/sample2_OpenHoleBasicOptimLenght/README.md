# Sample 2: Calculating optimal crack length and critical factor for a crack onset according to the coupled criterion

This sample shows how the code is able to calculate the critical load factor by minimizing the required load factor, using as an argument only the crack length at onset. The code use a ver basic Optimization Algorithm without being optimized for this particular application. Some modifications that can be employed could be the separation of the surrogate model differently for the stress criterion and for the energy criterion because are smooth functions, contrarily to the max of the two that is not smooth. 

## Usage

1. Copy FFMcracking.py and modOptimization.py from the src folder to the same folder where example2.py is located.
2. Edit example2.py, including the work directorty, computation directory, material properties, L range, and (not necessary)
 edit the number of iterations, etc.
3. Run 'abaqus cae noGui=example2.py'
4. Once finished, the result should be in the output defined, screen, or abaqus.rpy file.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[APACHE 2.0](https://www.apache.org/licenses/LICENSE-2.0)