NVCC = nvcc
NVCCFLAGS = --std=c++14 --expt-extended-lambda -O3 -rdc=true --generate-line-info
INC = -I/usr/include -I./gunrock -I./gunrock/externals/moderngpu/src -I./gunrock/externals/rapidjson/include
#INC = -I../../gunrock -I../../gunrock/externals/rapidjson/include -I/usr/include -I../../gunrock/externals/moderngpu/include -I../../gunrock/externals/moderngpu/src
#INC = -I../../gunrock -I../../gunrock/externals/rapidjson/include -I/usr/include -I../../gunrock/externals/cub -I../../gunrock/externals/moderngpu/include -I../../gunrock/externals/moderngpu/src
LINK = -Xcompiler -fopenmp -Xlinker -lgomp
#LINK = -Xcompiler -DBOOST_FOUND -Xlinker -lboost_system -Xlinker -lboost_chrono -Xlinker -lboost_timer -Xlinker -lboost_filesystem -Xcompiler -fopenmp -Xlinker -lgomp

GIT_SHA = -DGIT_SHA1="\"$(shell git rev-parse HEAD)\""
GUNROCK_GIT_SHA = -DGUNROCK_GIT_SHA1="\"$(shell cd gunrock; git rev-parse HEAD)\""

GUNROCK_SOURCES = gunrock/gunrock/util/str_to_T.cu \
		  gunrock/gunrock/util/test_utils.cu \
		  gunrock/gunrock/util/error_utils.cu

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM80 = -gencode=arch=compute_80,code=\"sm_80,compute_80\" # Ampere A100
GEN_SM75 = -gencode=arch=compute_75,code=\"sm_75,compute_75\" # Turing RTX20XX
GEN_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\" # Volta V100, Titan V
GEN_SM61 = -gencode=arch=compute_61,code=\"sm_61,compute_61\" # Pascal GTX10XX, Titan Xp
GEN_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\" # Pascal P100
GEN_SM52 = -gencode=arch=compute_52,code=\"sm_52,compute_52\" # Maxwell M40, M60, GTX9XX
GEN_SM50 = -gencode=arch=compute_50,code=\"sm_50,compute_50\" # Maxwell M10
GEN_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\" # Kepler K80
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" # Kepler K20, K40
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" # Kepler K10

# Note: Some of the architectures don't support Gunrock's
# RepeatFor (Cooperative Groups), e.g: SM35

# Select which architecture(s) to compile for:
SM_TARGETS = $(GEN_SM70)

#GPU code:
#-----------------
GPU_SOURCES = GPUTestMaxClique.cu cliqueMerging.cu
GPU_EXECUTABLE = test.o
gpu: $(GPU_EXECUTABLE)

#GPU validation code:
#-----------------
GPU_CORRECTNESS_SOURCES = GPUCorrectnessTest.cu cliqueMerging.cu
GPU_CORRECTNESS_EXECUTABLE = correctnessTest.o
correct_gpu: $(GPU_CORRECTNESS_EXECUTABLE)


all: $(GPU_EXECUTABLE) $(GPU_CORRECTNESS_EXECUTABLE)

$(GPU_EXECUTABLE):
	$(NVCC) $(NVCCFLAGS) $(INC) $(LINK) $(GIT_SHA) $(GUNROCK_GIT_SHA) $(GPU_SOURCES) $(GUNROCK_SOURCES) $(SM_TARGETS) -o $@


$(GPU_CORRECTNESS_EXECUTABLE):
	$(NVCC) $(NVCCFLAGS) $(INC) $(LINK) $(GIT_SHA) $(GUNROCK_GIT_SHA) $(GPU_CORRECTNESS_SOURCES) $(GUNROCK_SOURCES) $(SM_TARGETS) -o $@

clean:
	rm -f $(GPU_EXECUTABLE) $(GPU_CORRECTNESS_EXECUTABLE) 
