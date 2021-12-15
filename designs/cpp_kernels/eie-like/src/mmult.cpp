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

Kernel Description :

	This kernel is a systolic array based matrix multiplication. Though the
	maximum size of the input matrices are restricted to a smaller MAX_SIZE, it
	is still possible to use this approach and get better performance for larger
	matrices by using tiling.

	Arguments :

		int *a	 (input )  --> Input  Matrix A
		int *b	 (input )  --> Input  Matrix B
		int *c	 (output)  --> Output Matrix
		int  m_dim (input )  --> Row Size Matrix A
		int  k_dim (input )  --> Col Size Matrix A
		int  n_dim (input )  --> Col Size Matrix B

*******************************************************************************/

#include <stdio.h>

// Input Matrix Size - M dim
#define MATRIX_SIZE_M 48

// Input Matrix Size - K dim
#define MATRIX_SIZE_K 48

// Input Matrix Size - N dim
#define MATRIX_SIZE_N 24

// Maximum Array Size 
#define STORAGE_M_DIM 48
#define STORAGE_N_DIM 48
#define STORAGE_K_DIM 48
#define STORAGE_MK_NNZ 306

// Density percentage
#define MK_NNZ 306

// Parallel MAC Units
#define NUM_MACS 16

// TRIPCOUNT identifier
const unsigned int m_size = MATRIX_SIZE_M;
const unsigned int k_size = MATRIX_SIZE_K;
const unsigned int n_size = MATRIX_SIZE_N;
const unsigned int num_macs = NUM_MACS;

const unsigned int nz_size = MK_NNZ;
const unsigned int nz_size_vec = (int)(MK_NNZ/m_size);

extern "C" {
void mmult(const int* a_ptr, // Read-Only Matrix A
		   const int* a_idx, // Read-Only Matrix A
		   const int* a_val, // Read-Only Matrix A
		   const int* b, // Read-Only Matrix B
		   int* o,	   // Output Result
		   int m_dim,	// Matrix A Row Size
		   int k_dim,	// Matrix A Col Size
		   int n_dim,	 // Matrix B Col Size
		   int num_nz	// number of nonzeros
		   ) {
	
	// Local memory to store input and output matrices
	int localA_ptr[STORAGE_M_DIM+1];
//#pragma HLS ARRAY_PARTITION variable = localA_ptr dim = 0 complete

	int localA_idx[STORAGE_MK_NNZ]; 
//#pragma HLS ARRAY_PARTITION variable = localA_idx dim = 0 complete
	
	int localA_val[STORAGE_MK_NNZ]; 
//#pragma HLS ARRAY_PARTITION variable = localA_val dim = 0 complete

	int localB[STORAGE_K_DIM][STORAGE_N_DIM];
//#pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete

	int localO[STORAGE_M_DIM][STORAGE_N_DIM];
//#pragma HLS ARRAY_PARTITION variable = localO dim = 0 complete

// Burst reads on input matrices from global memory
// Read Input A metadata
// Auto-pipeline is going to apply pipeline to these loops
readAptr:
	for (int i = 0; i <= m_dim; i++) {
	#pragma HLS LOOP_TRIPCOUNT min = m_size+1 max = m_size+1
		localA_ptr[i] = a_ptr[i];
	}

readAidx:
	for (int i = 0; i < num_nz; i++) {
	#pragma HLS LOOP_TRIPCOUNT min = nz_size max = nz_size
		localA_idx[i] = a_idx[i];
	}

readAval:
	for (int i = 0; i < num_nz; i++) {
	#pragma HLS LOOP_TRIPCOUNT min = nz_size max = nz_size
		localA_val[i] = a_val[i];
	}

// Read Input B
readB:
	int loc = 0;
	for (int k = 0; k < k_dim; k++) {
	#pragma HLS LOOP_TRIPCOUNT min = k_size max = k_size
		for (int n = 0; n < n_dim; n++) {
			#pragma HLS LOOP_TRIPCOUNT min = n_size max = n_size
			localB[k][n] = b[loc];
			loc++;
		}
	}


// Perform SpMM (UmCk(A)-UkUn(B))
loop_m_o:
	for (int m_o = 0; m_o < (m_dim+num_macs-1)/num_macs; m_o++) {
	#pragma HLS LOOP_TRIPCOUNT min = m_size/num_macs max = m_size/num_macs
	loop_n:
		for (int n = 0; n < n_dim; n++) {
		#pragma HLS LOOP_TRIPCOUNT min = n_size max = n_size
		loop_m_i:
			for (int m_i = 0; m_i < num_macs; m_i++) {
			#pragma HLS unroll factor = num_macs
			
			int m = m_o*num_macs+m_i;
			
			loop_a_ptr:
				for (int z = localA_ptr[m], flag = 0; z < localA_ptr[m+1]; z++, flag++) {
				#pragma HLS LOOP_TRIPCOUNT min = nz_size_vec max = nz_size_vec
				
					
					#pragma HLS PIPELINE II=1 enable_flush rewind
					int a_cid = localA_idx[z];
					int a_val = localA_val[z];
					
					// get previous sum
					int last = (flag == 0) ? 0 : localO[m][n];

					int temp1, temp2;
					
					//#pragma HLS BIND_OP variable=temp1 op=mul impl=dsp
					temp1 = a_val * localB[a_cid][n];
					
					//#pragma HLS BIND_OP variable=temp2 op=add impl=dsp
					temp2 = last + temp1;

					// Write back results	
					localO[m][n] = temp2;

					#pragma HLS dependence variable=localO false 
				}
			}
		}
	}

	
// Burst write from output matrices to global memory
// Burst write from matrix C
writeO:
	loc = 0;
	for (int m = 0; m < m_dim; m++) {
	#pragma HLS LOOP_TRIPCOUNT min = m_size max = m_size
		for (int n = 0; n < n_dim; n++) {
			#pragma HLS LOOP_TRIPCOUNT min = n_size max = n_size
			o[loc] = localO[m][n];
			loc++;
		}
	}
}
}
