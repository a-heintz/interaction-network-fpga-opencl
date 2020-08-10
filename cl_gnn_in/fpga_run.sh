#!/bin/bash

unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA
unset CL_CONTEXT_EMULATOR_DEVICE_ALTERA

cd bin
# batch size of 10
./gnn_fpga

