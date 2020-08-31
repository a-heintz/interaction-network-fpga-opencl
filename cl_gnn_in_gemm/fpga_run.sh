#!/bin/bash

unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA
unset CL_CONTEXT_EMULATOR_DEVICE_ALTERA

aocl program acl0 gnn.aocx
aocl program acl1 gnn.aocx

cd bin
./gnn_fpga /tigress/aheintz/data/model_weights_LP_5.hdf5 /tigress/aheintz/data/test_LP_5.hdf5
