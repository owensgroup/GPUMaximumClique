# GPUMaximumClique
A maximum clique solver for GPUs

I hope to update the code to use the most recent release of gunrock, but for now, these steps will get the necessary externals, using the older versions used for the paper results:
```
git clone https://github.com/gunrock/gunrock.git
cd gunrock
git checkout 9aad94e8d9992a083e939c198337d084322226f3
cd externals
git clone https://github.com/moderngpu/moderngpu.git
cd moderngpu
git checkout 2b3985541c8e88a133769598c406c33ddde9d0a5
cd ..
git clone https://github.com/Tencent/rapidjson.git
```

Now you should be able to build using `make all`

To test for the brock200\_2.mtx dataset:
`./correctnessTest.o`

To run on your graph (in matrix market format):
`./test.o market [your_file]`

commandLine.md describes all of the command line parameter options that can be used to select different configuration options for the maximum clique solver.
