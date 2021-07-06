# Deploying an Interaction Network for Particle Track Reconstruction on FPGAs using OpenCL 

Associated Publication:

Heintz, A., Razavimaleki, V., Duarte, J., DeZoort, G., Ojalvo, I., Thais, S., Atkinson, M., Neubauer, M., Gray, L., Jindariani, S., Tran, N., Harris, P., Rankin, D., Aarrestad, T., Loncar, V., Pierini, M., Summers, S., Ngadiuba, J., Liu, M., Kreinar, E., Wu, Z., “Accelerated Charged Particle Tracking with Graph Neural Networks on FPGAs,” at Machine Learning and the Physical Sciences Workshop at the 34th Conference on Neural Information Processing Systems (NeurIPS), Dec. 11 2020.



This repository is an exploratory implementation of an Interaction Network (IN) for particle track reconstruction on FPGAs using OpenCL.

The network is described in the following repository: https://github.com/savvy379/princeton_gnn_tracking

Before running the implementation code on the FPGA, the input data for the neural network needs to be converted into the HDF5 format. The process_data.py file inside the "interaction_network" folder does this. As an example, the following runs the format conversion for a 5 GeV Pt cut data set from inside the "interaction_network" folder:
```
python process_data.py ./configs/train_IN_LP_5.yaml
```
Data paths need to be re-set inside each config file in the "configs" folder. Additionally, several slurm scripts contain procedures for training the IN and re-formatting data through the slurm workload manager. For more information on training the IN, refer to the network's base repository given above.

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
