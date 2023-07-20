## Compilation and Execution instructions
To compile the project use the CMakeLists.txt file provided and build it using make.
The compilation will produce an executable called "main".

The project to execute correctly requires an input dataset.
This input dataset must have a structure similar to the one of the test set provided as sample.
Hence each input tray must be labeled as "trayN" and inside each tray directory input images and 
ground truth mask/labels must be provided using the same conventions definet in the sample test set.
The sample testset has been included inside the "inputs" subdirectory.

To execute the code run the executable specifying as first parameter the path to the input dataset
    ./main <base-path-to-input-trays>

To run on the sample testset provided with the homework hence do:
    ./main ../inputs/Food_leftover_dataset/


## Output description
The code will sequentially process each tray on the input dataset, for each one an output image is shown and once the last 
tray has been processed the program will stall on waitKey(0).

For each tray the computed metrics, food masks and bounding boxes are stored inside its directory under the subdirectories 
"output_bounding_boxes" and "output_masks" created by the program.

Output bounding boxes and food mask follow the same conventions used in the sample test set provided.