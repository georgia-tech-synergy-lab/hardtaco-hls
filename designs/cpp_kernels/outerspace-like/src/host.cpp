/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*	 http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/*******************************************************************************

Description:

	This is a matrix multiplication which showcases the "Systolic Array" based
	algorithm design. Systolic array type of implementation is well suited for
	FPGAs. It is a good coding practice to convert base algorithm into Systolic
	Array implementation if it is feasible to do so.

*******************************************************************************/
#include "xcl2.hpp"
#include "cmdlineparser.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <bits/stdc++.h>
#include "timer.h"


using namespace sda::utils;
using namespace std;

// Input Matrix Size - M dim
#define MATRIX_SIZE_M 48

// Input Matrix Size - K dim
#define MATRIX_SIZE_K 48

// Input Matrix Size - N dim
#define MATRIX_SIZE_N 24

// Maximum Array Size 
#define MAX_SIZE 48

// Density percentage
#define MK_NNZ 306
#define KN_NNZ 550


// Read Input Files and Save
vector<string> read_inputs(string filename) {
	ifstream fin;
	string line;
	vector<string> v;
	fin.open(filename);
	while(!fin.eof()){
		fin>>line;
		//cout<<line<<" ";
		stringstream ss(line);

		while(ss.good()) {
			string substr;
			getline(ss, substr, ',');
			v.push_back(substr);
		}
	}

	return v;
}


// Software implementation of Matrix Multiplication
// The inputs are of the size (DATA_SIZE x DATA_SIZE)
void m_softwareGold(std::vector<int, aligned_allocator<int> >& in1, // Input Matrix 1
					std::vector<int, aligned_allocator<int> >& in2, // Input Matrix 2
					std::vector<int, aligned_allocator<int> >& out  // Output Matrix
					) {
	// Perform Matrix multiply Out = In1 x In2
	for (int i = 0; i < MATRIX_SIZE_M; i++) {
		for (int j = 0; j < MATRIX_SIZE_N; j++) {
			for (int k = 0; k < MATRIX_SIZE_K; k++) {
				out[i * MATRIX_SIZE_N + j] += in1[i * MATRIX_SIZE_K + k] * in2[k * MATRIX_SIZE_N + j];
			}
		}
	}
}


/******************************************************************

						   MAIN() FUNCTION

 *****************************************************************/
int main(int argc, char** argv) {
	
	// Command Line Parser
	CmdLineParser parser;

	// Switches
	//**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
	parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
	parser.addSwitch("--input_matrix_a_val_file", "-av", "input matrix a value test data file", "");
	parser.addSwitch("--input_matrix_a_idx_file", "-ai", "input matrix a idx test data file", "");
	parser.addSwitch("--input_matrix_a_ptr_file", "-ap", "input matrix a ptr test data file", "");
	parser.addSwitch("--input_matrix_b_val_file", "-bv", "input matrix b value test data file", "");
	parser.addSwitch("--input_matrix_b_idx_file", "-bi", "input matrix b idx test data file", "");
	parser.addSwitch("--input_matrix_b_ptr_file", "-bp", "input matrix b ptr test data file", "");
	parser.addSwitch("--input_cfg_file", "-f", "input config file", "");   
	parser.addSwitch("--output_golden_file", "-g", "Compare File to compare result", "");
	parser.parse(argc, argv);


	// Read settings
	std::string binaryFile = parser.value("xclbin_file");
	std::string matrixAvalfile = parser.value("input_matrix_a_val_file");
	std::string matrixAidxfile = parser.value("input_matrix_a_idx_file");
	std::string matrixAptrfile = parser.value("input_matrix_a_ptr_file");
	std::string matrixBvalfile = parser.value("input_matrix_b_val_file");
	std::string matrixBidxfile = parser.value("input_matrix_b_idx_file");
	std::string matrixBptrfile = parser.value("input_matrix_b_ptr_file");
	std::string cfgfile = parser.value("input_cfg_file");
	std::string goldenfile = parser.value("output_golden_file");

	if (argc < 10) {
		parser.printHelp();
		return EXIT_FAILURE;
	}

	// Allocate Memory in Host Memory
	if (MATRIX_SIZE_M > MAX_SIZE || MATRIX_SIZE_K > MAX_SIZE || MATRIX_SIZE_N > MAX_SIZE) {
		std::cout << "Size is bigger than internal buffer size, please use a "
					 "size smaller than "
				  << MAX_SIZE << "!" << std::endl;
		return EXIT_FAILURE;
	}
	
	size_t mk_matrix_size = MATRIX_SIZE_M * MATRIX_SIZE_K;
	size_t mk_matrix_size_bytes = sizeof(int) * mk_matrix_size;
	size_t mk_val_size = MK_NNZ;
	size_t mk_val_size_bytes = sizeof(int) * mk_val_size;
	size_t mk_idx_size = MK_NNZ;
	size_t mk_idx_size_bytes = sizeof(int) * mk_idx_size;
	size_t mk_ptr_size = MATRIX_SIZE_K + 1;
	size_t mk_ptr_size_bytes = sizeof(int) * mk_ptr_size;

	size_t kn_matrix_size = MATRIX_SIZE_K * MATRIX_SIZE_N;
	size_t kn_matrix_size_bytes = sizeof(int) * kn_matrix_size;
	size_t kn_val_size = KN_NNZ;
	size_t kn_val_size_bytes = sizeof(int) * kn_val_size;
	size_t kn_idx_size = KN_NNZ;
	size_t kn_idx_size_bytes = sizeof(int) * kn_idx_size;
	size_t kn_ptr_size = MATRIX_SIZE_K + 1;
	size_t kn_ptr_size_bytes = sizeof(int) * kn_ptr_size;

	size_t mn_matrix_size = MATRIX_SIZE_M * MATRIX_SIZE_N;
	size_t mn_matrix_size_bytes = sizeof(int) * mn_matrix_size;	
	
	
	cl_int err;
	cl::CommandQueue q;
	cl::Context context;
	cl::Kernel krnl_systolic_array;
	
	std::vector<int, aligned_allocator<int> > source_in1_val(mk_val_size);
	std::vector<int, aligned_allocator<int> > source_in1_idx(mk_idx_size);
	std::vector<int, aligned_allocator<int> > source_in1_ptr(mk_ptr_size);
	
	std::vector<int, aligned_allocator<int> > source_in2_val(kn_val_size);
	std::vector<int, aligned_allocator<int> > source_in2_idx(kn_idx_size);
	std::vector<int, aligned_allocator<int> > source_in2_ptr(kn_ptr_size);

	std::vector<int, aligned_allocator<int> > source_in1(mk_matrix_size);
	std::vector<int, aligned_allocator<int> > source_in2(kn_matrix_size);
	std::vector<int, aligned_allocator<int> > source_hw_results(mn_matrix_size);
	std::vector<int, aligned_allocator<int> > source_sw_results(mn_matrix_size);
	
	// Create SW and HW Result Matrices
	for (size_t i = 0; i < mn_matrix_size; i++) {
		source_sw_results[i] = 0;
		source_hw_results[i] = 0;
	}
	
	
	// Read in source_in1 input matrix A
	vector<string> v_A_val;
	v_A_val = read_inputs(matrixAvalfile);

	if (mk_val_size == v_A_val.size()) {
		for (size_t i =0 ; i < v_A_val.size(); i++)
			source_in1_val[i] = stoi(v_A_val[i]);
			//cout << v_A[i] << endl;
	} else {
		std::cout << "Input MK VAL File Read Size Mismatch" << std::endl;
		std::cout << "v_A_val.size: " << v_A_val.size() << std::endl;
		std::cout << "mk_val_size: " << mk_val_size << std::endl;
		return EXIT_FAILURE;	   
	}	
	
	vector<string> v_A_idx;
	v_A_idx = read_inputs(matrixAidxfile);

	if (mk_idx_size == v_A_idx.size()) {
		for (size_t i =0 ; i < v_A_idx.size(); i++)
			source_in1_idx[i] = stoi(v_A_idx[i]);
			//cout << v_A[i] << endl;
	} else {
		std::cout << "Input MK IDX File Read Size Mismatch" << std::endl;
		std::cout << "v_A_idx.size: " << v_A_idx.size() << std::endl;
		std::cout << "mk_idx_size: " << mk_idx_size << std::endl;
		return EXIT_FAILURE;	   
	}	
	
	vector<string> v_A_ptr;
	v_A_ptr = read_inputs(matrixAptrfile);

	if (mk_ptr_size == v_A_ptr.size()) {
		for (size_t i =0 ; i < v_A_ptr.size(); i++)
			source_in1_ptr[i] = stoi(v_A_ptr[i]);
			//cout << v_A[i] << endl;
	} else {
		std::cout << "Input MK PTR File Read Size Mismatch" << std::endl;
		std::cout << "v_A_ptr.size: " << v_A_ptr.size() << std::endl;
		std::cout << "mk_ptr_size: " << mk_ptr_size << std::endl;
		return EXIT_FAILURE;	   
	}	
	
	// Read in source_in2 input matrix B
	vector<string> v_B_val;
	v_B_val = read_inputs(matrixBvalfile);

	if (kn_val_size == v_B_val.size()) {
		for (size_t i =0 ; i < v_B_val.size(); i++)
			source_in2_val[i] = stoi(v_B_val[i]);
	} else {
		std::cout << "Input KN VAL File Read Size Mismatch" << std::endl;
		std::cout << "v_B_val.size: " << v_B_val.size() << std::endl;
		std::cout << "kn_val_size: " << kn_val_size << std::endl;
		return EXIT_FAILURE;	   
	}	
	
	vector<string> v_B_idx;
	v_B_idx = read_inputs(matrixBidxfile);

	if (kn_idx_size == v_B_idx.size()) {
		for (size_t i =0 ; i < v_B_idx.size(); i++)
			source_in2_idx[i] = stoi(v_B_idx[i]);
	} else {
		std::cout << "Input KN IDX File Read Size Mismatch" << std::endl;
		std::cout << "v_B_idx.size: " << v_B_idx.size() << std::endl;
		std::cout << "kn_idx_size: " << kn_idx_size << std::endl;
		return EXIT_FAILURE;	   
	}	
	
	vector<string> v_B_ptr;
	v_B_ptr = read_inputs(matrixBptrfile);

	if (kn_ptr_size == v_B_ptr.size()) {
		for (size_t i =0 ; i < v_B_ptr.size(); i++)
			source_in2_ptr[i] = stoi(v_B_ptr[i]);
	} else {
		std::cout << "Input KN PTR File Read Size Mismatch" << std::endl;
		std::cout << "v_B_ptr.size: " << v_B_ptr.size() << std::endl;
		std::cout << "kn_ptr_size: " << kn_ptr_size << std::endl;
		return EXIT_FAILURE;	   
	}
	
	// Read in configuration file
	// Configuration parameter (M, N, K, MK_NNZ, KN_NNZ)
	vector<string> v_cfg;
	v_cfg = read_inputs(cfgfile);

	int m_dim = 0;
	int n_dim = 0;
	int k_dim = 0;
	int mk_nnz = 0;
	int kn_nnz = 0;

	if (v_cfg.size() == 5) {
		m_dim = stoi(v_cfg[0]);
		n_dim = stoi(v_cfg[1]);
		k_dim = stoi(v_cfg[2]);
		mk_nnz = stoi(v_cfg[3]);
		kn_nnz = stoi(v_cfg[4]);
	} else {
		std::cout << "Config parameters missing..." << std::endl;
		std::cout << "(M, N, K, MK_NNZ, KN_NNZ)"  << std::endl;
		return EXIT_FAILURE;	   
	}
	
	// Read in golden output
	vector<string> v_O;
	v_O = read_inputs(goldenfile);

	if (mn_matrix_size == v_O.size()) {
		for (size_t i =0 ; i < v_O.size(); i++)
			source_sw_results[i] = stoi(v_O[i]);
	} else {
		std::cout << "Golden File Read Size Mismatch" << std::endl;
		std::cout << "v_O.size: " << v_O.size() << std::endl;
		std::cout << "mn_matrix_size: " << mn_matrix_size << std::endl;
		return EXIT_FAILURE;	   
	}

	

	// OPENCL HOST CODE AREA START
	auto devices = xcl::get_xil_devices();

	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	for (unsigned int i = 0; i < devices.size(); i++) {
		auto device = devices[i];
		// Creating Context and Command Queue for selected Device
		OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
		} else {
			std::cout << "Device[" << i << "]: program successful!\n";
			OCL_CHECK(err, krnl_systolic_array = cl::Kernel(program, "mmult", &err));
			valid_device = true;
			break; // we break because we found a valid device
		}
	}
	if (!valid_device) {
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_in1_ptr(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, mk_ptr_size_bytes,
										 source_in1_ptr.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_in1_idx(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, mk_idx_size_bytes,
										 source_in1_idx.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_in1_val(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, mk_val_size_bytes,
										 source_in1_val.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_in2_ptr(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, kn_ptr_size_bytes,
										 source_in2_ptr.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_in2_idx(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, kn_idx_size_bytes,
										 source_in2_idx.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_in2_val(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, kn_val_size_bytes,
										 source_in2_val.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, mn_matrix_size_bytes,
											source_hw_results.data(), &err)); 


	OCL_CHECK(err, err = krnl_systolic_array.setArg(0, buffer_in1_ptr));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(1, buffer_in1_idx));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(2, buffer_in1_val));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(3, buffer_in2_ptr));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(4, buffer_in2_idx));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(5, buffer_in2_val));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(6, buffer_output));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(7, m_dim));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(8, k_dim));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(9, n_dim));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(10, mk_nnz));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(11, kn_nnz));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1_ptr, buffer_in1_idx, buffer_in1_val, buffer_in2_ptr, buffer_in2_idx, buffer_in2_val}, 0 /* 0 means from host*/));

	// Launch the Kernel
	TIMER_INIT(1);
	TIMER_START(0);
	OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array));
	q.finish();
	TIMER_STOP_ID(0);
	printf("------------------------------------------------------\n");
	printf("Kernel Time : %12.4f ms\n", TIMER_REPORT_MS(0));
	printf("------------------------------------------------------\n");
	
	
	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();
	// OPENCL HOST CODE AREA END

	// Compute Software Results
	// m_softwareGold(source_in1, source_in2, source_sw_results);
	
	// Compare the results of the Device to the simulation
	int match = 0;
	for (int i = 0; i < MATRIX_SIZE_M * MATRIX_SIZE_N; i++) {
		if (source_hw_results[i] != source_sw_results[i]) {
			std::cout << "Error: Result mismatch" << std::endl;
			std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
					  << " Device result = " << source_hw_results[i] << std::endl;
			match = 1;
			break;
		}
	}

	std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
	return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}
