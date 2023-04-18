LD_LIBRARY_PATH=build  ./build/examples/cpu-alex 2 & 
LD_LIBRARY_PATH=build  ./build/examples/cpu-gg 6 &
LD_LIBRARY_PATH=build  ./build/examples/graph_squeezenet --target=cl
