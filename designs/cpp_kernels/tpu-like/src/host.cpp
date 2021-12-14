/**
* Copyright (C) 2019-2021 Xilinx, Inc
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
	parser.addSwitch("--input_matrix_a_file", "-a", "input matrix a test data file", "");
	parser.addSwitch("--input_matrix_b_file", "-b", "input matrix b test data file", "");
	parser.addSwitch("--input_cfg_file", "-f", "input config file", "");   
	parser.addSwitch("--output_golden_file", "-g", "Compare File to compare result", "");
	parser.parse(argc, argv);


	// Read settings
	std::string binaryFile = parser.value("xclbin_file");
	std::string matrixAfile = parser.value("input_matrix_a_file");
	std::string matrixBfile = parser.value("input_matrix_b_file");
	std::string cfgfile = parser.value("input_cfg_file");
	std::string goldenfile = parser.value("output_golden_file");

	if (argc < 6) {
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

	size_t kn_matrix_size = MATRIX_SIZE_K * MATRIX_SIZE_N;
	size_t kn_matrix_size_bytes = sizeof(int) * kn_matrix_size;

	size_t mn_matrix_size = MATRIX_SIZE_M * MATRIX_SIZE_N;
	size_t mn_matrix_size_bytes = sizeof(int) * mn_matrix_size;

	cl_int err;
	cl::CommandQueue q;
	cl::Context context;
	cl::Kernel krnl_systolic_array;

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
	vector<string> v_A;
	v_A = read_inputs(matrixAfile);

	if (mk_matrix_size == v_A.size()) {
		for (size_t i =0 ; i < v_A.size(); i++)
			source_in1[i] = stoi(v_A[i]);
			//cout << v_A[i] << endl;
	} else {
		std::cout << "Input MK File Read Size Mismatch" << std::endl;
		std::cout << "v_A.size: " << v_A.size() << std::endl;
		std::cout << "mk_matrix_size: " << mk_matrix_size << std::endl;
		return EXIT_FAILURE;	   
	}

	// Read in source_in2 input matrix B
	vector<string> v_B;
	v_B = read_inputs(matrixBfile);

	if (kn_matrix_size == v_B.size()) {
		for (size_t i =0 ; i < v_B.size(); i++)
			source_in2[i] = stoi(v_B[i]);
	} else {
		std::cout << "Input KN File Read Size Mismatch" << std::endl;
		std::cout << "v_B.size: " << v_B.size() << std::endl;
		std::cout << "kn_matrix_size: " << kn_matrix_size << std::endl;
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
		OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, nullptr, &err);
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
	OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, mk_matrix_size_bytes,
										 source_in1.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, kn_matrix_size_bytes,
										 source_in2.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, mn_matrix_size_bytes,
											source_hw_results.data(), &err));


	OCL_CHECK(err, err = krnl_systolic_array.setArg(0, buffer_in1));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(1, buffer_in2));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(2, buffer_output));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(3, m_dim));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(4, k_dim));
	OCL_CHECK(err, err = krnl_systolic_array.setArg(5, n_dim));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

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
