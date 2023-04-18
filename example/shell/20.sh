LD_LIBRARY_PATH=build  ./build/examples/cpu-alex 7 & 
LD_LIBRARY_PATH=build  ./build/examples/cpu-gg 1 &
LD_LIBRARY_PATH=build  ./build/examples/graph_squeezenet --target=cl
