# ZeroWaste
Zero Waste is a computer vision system able to segment and classify food trays coming from our university canteen and to estimate
food waste using classic computer vision techniques.<br>
Zero Waste analyzes two main images: one before the meal and one at the end of the meal. From the first image the system is able to
recognize the various types of food, keeping track of the initial quantity of each food. From the last image of the tray (the one taken at the end) the system
recognize which types of food are still present and in which quantity, providing a leftover estimation.<br>
To assess the system’s performance, Zero Waste implements the mean Average Precision (mAP) and the mean Intersection over Union (mIoU) metrics.

Zero Waste has been developed in C++ using the OpenCV library, for a full description of the system see the attached [report](https://github.com/FrancescoVisentin/ZeroWaste/blob/main/ZeroWaste%20Report.pdf).

![2](https://github.com/FrancescoVisentin/ZeroWaste/assets/74708171/1131728d-4427-4cec-95fb-83b0de1cff90)


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

To run on the sample testset provided with the homework hence do (assuming to be inside the build directory):
    ./main ../inputs/Food_leftover_dataset/


## Output description
The code will sequentially process each tray on the input dataset, for each one an output image is shown and once the last 
tray has been processed the program will stall on waitKey(0).

For each tray the computed metrics, food masks and bounding boxes are stored inside its directory under the subdirectories 
"output_bounding_boxes" and "output_masks" created by the program.

Output bounding boxes and food mask follow the same conventions used in the sample test set provided.
