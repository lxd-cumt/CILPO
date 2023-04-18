LD_LIBRARY_PATH=build  ./build/examples/cpu-alex 3 & 
LD_LIBRARY_PATH=build  ./build/examples/cpu-gg 5 &
LD_LIBRARY_PATH=build  ./build/examples/graph_squeezenet --target=cl
