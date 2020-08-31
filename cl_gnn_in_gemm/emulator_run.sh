#!/bin/bash
cd bin
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./gnn_fpga /tigress/aheintz/data/model_weights_LP_5.hdf5 /tigress/aheintz/data/test_LP_5.hdf5
