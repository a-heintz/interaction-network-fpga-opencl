#!/bin/bash

rm -rf bin

aoc -march=emulator -legacy-emulator device/gnn.cl -o bin/gnn.aocx

make
