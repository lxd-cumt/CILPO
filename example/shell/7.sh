LD_LIBRARY_PATH=build  ./build/examples/cpu-alex 1 & 
LD_LIBRARY_PATH=build  ./build/examples/graph_googlenet --target=cl &
LD_LIBRARY_PATH=build  ./build/examples/cpu-squ 7