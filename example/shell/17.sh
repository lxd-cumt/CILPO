LD_LIBRARY_PATH=build  ./build/examples/cpu-alex 4 & 
LD_LIBRARY_PATH=build  ./build/examples/cpu-gg 4 &
LD_LIBRARY_PATH=build  ./build/examples/graph_squeezenet --target=cl
