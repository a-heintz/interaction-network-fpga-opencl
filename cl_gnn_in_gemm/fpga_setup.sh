#!/bin/bash

unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA
unset CL_CONTEXT_EMULATOR_DEVICE_ALTERA

rm -rf bin

aoc device/gnn.cl -o bin/gnn.aocx -report
#aoc -rtl device/gnn.cl -report
make
