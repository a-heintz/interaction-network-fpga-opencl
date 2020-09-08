#!/bin/bash

unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA
unset CL_CONTEXT_EMULATOR_DEVICE_ALTERA

cd bin

aocl program acl0 gnn.aocx
aocl program acl1 gnn.aocx

echo "Pt: 5 GeV"
./gnn_fpga /tigress/aheintz/data/model_weights_LP_5.hdf5 /tigress/aheintz/data/test_LP_5 100
echo "Pt: 3 GeV"
./gnn_fpga /tigress/aheintz/data/model_weights_LP_3.hdf5 /tigress/aheintz/data/test_LP_3 100
echo "Pt: 1.5 GeV"
./gnn_fpga /tigress/aheintz/data/model_weights_LP_1p5.hdf5 /tigress/aheintz/data/test_LP_1p5 100
echo "Pt: 0.75 GeV"
./gnn_fpga /tigress/aheintz/data/model_weights_LP_0p75.hdf5 /tigress/aheintz/data/test_LP_0p75 60
#echo "Pt: 0.5 GeV"
#./gnn_fpga /tigress/aheintz/data/model_weights_LP_0p5.hdf5 /tigress/aheintz/data/test_LP_0p5
