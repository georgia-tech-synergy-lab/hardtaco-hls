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
#define STORAGE_KN_NNZ 550

// Density percentage
#define MK_NNZ 306
#define KN_NNZ 550

// Parallel MAC Units
#define NUM_MACS 16

// TRIPCOUNT identifier
const unsigned int m_size = MATRIX_SIZE_M;
const unsigned int k_size = MATRIX_SIZE_K;
const unsigned int n_size = MATRIX_SIZE_N;
const unsigned int num_macs = NUM_MACS;

const unsigned int mk_nz_size = MK_NNZ;
const unsigned int mk_nz_size_vec = (int)(MK_NNZ/m_size);
const unsigned int kn_nz_size = KN_NNZ;
const unsigned int kn_nz_size_vec = (int)(KN_NNZ/n_size);

extern "C" {
void mmult(const int* a_ptr, // Read-Only Matrix A
		   const int* a_idx, // Read-Only Matrix A
		   const int* a_val, // Read-Only Matrix A
		   const int* b_ptr, // Read-Only Matrix B
		   const int* b_idx, // Read-Only Matrix B
		   const int* b_val, // Read-Only Matrix B
		   int* o,	   // Output Result
		   int m_dim,	// Matrix A Row Size
		   int k_dim,	// Matrix A Col Size
		   int n_dim,	 // Matrix B Col Size
		   int mk_nnz,	// number of nonzeros
		   int kn_nnz
		   ) {

	// Local memory to store input and output matrices
	int localA_ptr[STORAGE_M_DIM+1];
//#pragma HLS ARRAY_PARTITION variable = localA_ptr dim = 0 complete

	int localA_idx[STORAGE_MK_NNZ]; // worst case allocation
//#pragma HLS ARRAY_PARTITION variable = localA_idx dim = 0 complete
	
	int localA_val[STORAGE_MK_NNZ]; // worst case allocation
//#pragma HLS ARRAY_PARTITION variable = localA_val dim = 0 complete

	int localB_ptr[STORAGE_N_DIM+1];
//#pragma HLS ARRAY_PARTITION variable = localB_ptr dim = 0 complete

	int localB_idx[STORAGE_KN_NNZ]; // worst case allocation
//#pragma HLS ARRAY_PARTITION variable = localB_idx dim = 0 complete
	
	int localB_val[STORAGE_KN_NNZ]; // worst case allocation
//#pragma HLS ARRAY_PARTITION variable = localB_val dim = 0 complete

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
	for (int i = 0; i < mk_nnz; i++) {
#pragma HLS LOOP_TRIPCOUNT min = mk_nz_size max = mk_nz_size
		localA_idx[i] = a_idx[i];
	}

readAval:
	for (int i = 0; i < mk_nnz; i++) {
#pragma HLS LOOP_TRIPCOUNT min = mk_nz_size max = mk_nz_size
		localA_val[i] = a_val[i];
	}

// Read Input B
readBptr:
	for (int i = 0; i <= n_dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = n_size+1 max = n_size+1
		localB_ptr[i] = b_ptr[i];
	}
	
readBidx:
	for (int i = 0; i < kn_nnz; i++) {
#pragma HLS LOOP_TRIPCOUNT min = kn_nz_size max = kn_nz_size
		localB_idx[i] = b_idx[i];
	}

readBval:
	for (int i = 0; i < kn_nnz; i++) {
#pragma HLS LOOP_TRIPCOUNT min = kn_nz_size max = kn_nz_size
		localB_val[i] = b_val[i];
	}
	

setzero:
	for (int m = 0; m < STORAGE_M_DIM; m++) {
		for (int n = 0; n < STORAGE_N_DIM; n++) {
			localO[m][n] = 0;
		}
	}
	

// Perform SpGEMM matrix multiply (UmCk(A)-UnCk(B))
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
				
				int kA = localA_ptr[m];
				int pA2_end = localA_ptr[(m + 1)];
				int kx = localB_ptr[n];
				int px2_end = localB_ptr[(n + 1)];

				bool flag = 0;
			loop3:
				while (kA < pA2_end && kx < px2_end) {
				#pragma HLS LOOP_TRIPCOUNT min = mk_nz_size_vec+kn_nz_size_vec max = mk_nz_size_vec+kn_nz_size_vec
				#pragma HLS PIPELINE II=1 enable_flush rewind
					int kA0 = localA_idx[kA];
					int kx0 = localB_idx[kx];
					int k;
					if (kA0 < kx0) {
						k = kA0;
					} else {
						k = kx0;
					}
					
					//int32_t n = TACO_MIN(nA0,nx0);
					if (kA0 == k && kx0 == k) {
					// get previous sum
						int last = (flag == 0) ? 0 : localO[m][n];
						flag = 1;
											
						// Write back results
						int temp1, temp2;
							
						//#pragma HLS BIND_OP variable=temp1 op=mul impl=dsp
						temp1 = localA_val[kA] * localB_val[kx];
							
						//#pragma HLS BIND_OP variable=temp2 op=add impl=dsp
						temp2 = last + temp1;
							
						localO[m][n] = temp2;

						//#pragma HLS dependence variable=localO false
					}
					kA += (int)(kA0 == k);
					kx += (int)(kx0 == k);
				}
			}
		}
	}


// Burst write from output matrices to global memory
// Burst write from matrix C
writeO:
	int loc = 0;
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
