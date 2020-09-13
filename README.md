# interaction_network_fpga_opencl

This repository is an exploratory implementation of an Interaction Network for particle track reconstruction on FPGAs using OpenCL.

The network is described in the following repository: https://github.com/savvy379/princeton_gnn_tracking

Before running the implementation code on the FPGA, the input data for the neural network needs to be converted into the HDF5 format. The process_data.py file inside the "interaction_network" folder does this. The following runs the format conversion for a 5 GeV Pt cut data set from inside the "interaction_network" folder:
```
python process_data.py ./configs/train_IN_LP_5.yaml
```
Data paths need to be re-set inside each config file in the "configs" folder.

Several bash scripts are provided to easily set up the necessary environment and compile the code on the FPGAs. The only change that needs to be made is to re-set data and local path locations inside each script.

To set up the OpenCL environment (must be done before compiling or running the OpenCL implementation code), run the following:
```
source environmentSetUp.sh
```
Because it takes hours to compile for an FPGA, it is recommended to first compile the code on an emulator. To compile the code on the emulator, run the emulator_setup.sh script:
```
bash emulator_setup.sh
```
To run the code on the emulator, run the emulator_run.sh script:
```
bash emulator_run.sh
```
To compile the code on the fpga, run the fpga_setup.sh script:
```
bash fpga_setup.sh
```
To run the code on the emulator, run the fpga_run.sh script:
```
bash fpga_run.sh
```
