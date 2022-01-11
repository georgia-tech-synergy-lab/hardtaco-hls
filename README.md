----------------------------------------------------------------------------------------------------------
Hard TACO
----------------------------------------------------------------------------------------------------------
This project enables users to utilize various TACO output templates to generate
hardware designs for various sparse accelerators. 

Helpful & Relevant Links:
	1) Vitis Accel Examples: https://github.com/Xilinx/Vitis_Accel_Examples
	
	2) SuiteSparse Matrix Collection: https://sparse.tamu.edu/
	
	3) TACO compiler: http://tensor-compiler.org/

----------------------------------------------------------------------------------------------------------
Directory structure
----------------------------------------------------------------------------------------------------------
	- sim_init.py: script to generate paramateried HLS and testbench
	- sim_param.cfg: configuration file (e.g. number of PEs)
	- designs: HLS cpp code using hand tuned TACO generated outputs
		-commons: Common code (includes etc.)	
		-cpp_kernels: HLS cpp code directory (host.cpp is testbench, mmult.cpp is HLS kernel) 
			-tpu_like: TPU-like accelerator HLS implementation 
				(GEMM with UmUk-UkUn compression)
			-eie_like: EIE-like accelerator HLS implementation 
				(SpMM with UmCk-UkUn compression)
			-extensor_like: ExTensor-like accelerator HLS implementation 
				(SpGEMM Inner Product with UmCk-UnCk compression)
			-outerspace_like: OuterSPACE-like accelerator HLS implementation 
				(SpGEMM Outer Product with UkCm-UkCn compression)
			-matraptor_like: MatRaptor-like accelerator HLS implementation 
				(SpGEMM Col-wise Product with UkCm-UnCk compression)

	- workloads: Contains sparse matrices and scripts for testbench generation
		- gen_tenstbench.py: generate testbench files with real or random matrices
		- gen_format.py: helper functions for compression format conversions
		- matrix_cfg
			- *.cfg: Workload script of "M,N,K,MK_NNZ,KN_NNZ"
		- suitesparse
			- * : different suitespace workloads (mtx files)

----------------------------------------------------------------------------------------------------------
Requirements:
----------------------------------------------------------------------------------------------------------
	- Xilinx Vitis HLS, Vivado, XRT Tools (Tested on version 2020.2 and xilinx_u50_gen3x16_xdma_201920_3)
	
----------------------------------------------------------------------------------------------------------
Changing/Adding Workload:
----------------------------------------------------------------------------------------------------------
	- Adding Real Matrix from SuiteSparse
		1) Find desired matrix from SuiteSparse.
			The matrix cannot be too big; otherwise, the FPGA cannot fit all data in
			memory. Tiling is needed if so.
		2) Add mtx directory of the desired matrix into ./workload/suitesparse
		3) Create new <matrix_name>.cfg file with the information: "M,N,K,MK_NNZ,KN_NNZ"
			The M, K, and MK_NNZ is specificed in SuiteSparse.
			N and KN_NNZ is up to the user.	
		4) In sim_init.py, enable the new real matrix by following previous examples.
			Use -u flag to specify whether it is undirected or directed
			Use -r real to specify it is using real data.
			
	- Customizing Random Matrix
		1) Update example.cfg file with the information: "M,N,K,MK_NNZ,KN_NNZ"

----------------------------------------------------------------------------------------------------------
Run examples
----------------------------------------------------------------------------------------------------------
	1) Update sim_param.cfg with desired configuration (make sure there is no new line at end)
	2) Run 'make sim' to generate sim directory
	3) Run 'cd sim/cpp_kernels/<*-like>' to enter desired HLS sparse accelerator configuration
	4) To run SW emulation
		make check TARGET=sw_emu DEVICE=xilinx_u50_gen3x16_xdma_201920_3 HOST_ARCH=x86
	5) To run HW emulation
		make check TARGET=hw_emu DEVICE=xilinx_u50_gen3x16_xdma_201920_3 HOST_ARCH=x86
	6) To run HW
		make check TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3 HOST_ARCH=x86
	7) HLS report and Verilog are found in _x.hw.xilinx_u50_gen3x16_xdma_201920_3/


----------------------------------------------------------------------------------------------------------
Known Issues/Comments:
----------------------------------------------------------------------------------------------------------
	- Large designs & workloads (especially using large buffer sizes) will cause sw_emu to crash.
	- Most likely HLS will not be able to generate large designs in a reasonable amount of time.
	
