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
#include "timer.h"

// Input Matrix Size - M dim
#define MATRIX_SIZE_M 48

// Input Matrix Size - K dim
#define MATRIX_SIZE_K 48

// Input Matrix Size - N dim
#define MATRIX_SIZE_N 24

// Maximum Array Size
#define MAX_SIZE 48

// Parallel MAC Units
#define NUM_MACS 16
#define NUM_MAC_X 4
#define NUM_MAC_Y 4


// TRIPCOUNT identifier
const unsigned int m_size = MATRIX_SIZE_M;
const unsigned int k_size = MATRIX_SIZE_K;
const unsigned int n_size = MATRIX_SIZE_N;
const unsigned int mac_x =  NUM_MAC_X;
const unsigned int mac_y = NUM_MAC_Y;

extern "C" {
void mmult(const int* a, // Read-Only Matrix A
		const int* b, // Read-Only Matrix B
		int* o,	   // Output Result
		int m_dim,	// Matrix A Row Size
		int k_dim,	// Matrix A Col Size
		int n_dim	 // Matrix B Col Size
		) {

	// Local memory to store input and output matrices
	int localA[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = localA dim = 0 complete

	int localB[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = localB dim = 0 complete

	int localO[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = localO dim = 0 complete

// Burst reads on input matrices from global memory
// Read Input A
// Auto-pipeline is going to apply pipeline to these loops
readA:
	int loc = 0;
	for (int m = 0; m < m_dim; m++) {
	#pragma HLS LOOP_TRIPCOUNT min = m_size max = m_size
		for (int k = 0; k < k_dim; k++) {
			#pragma HLS LOOP_TRIPCOUNT min = k_size max = k_size
			localA[m][k] = a[loc];
			loc++;
		}
	}

// Read Input B
readB:
	loc = 0;
	for (int k = 0; k < k_dim; k++) {
	#pragma HLS LOOP_TRIPCOUNT min = k_size max = k_size
		for (int n = 0; n < n_dim; n++) {
			#pragma HLS LOOP_TRIPCOUNT min = n_size max = n_size
			localB[k][n] = b[loc];
			loc++;
		}
	}

// Compute Core Logic 
loop_m_o:
	for(int m_o = 0; m_o < (m_dim+mac_x-1)/mac_x; m_o++) {
	#pragma HLS LOOP_TRIPCOUNT min = m_size/mac_x max = m_size/mac_x
	loop_n_o:
		for(int n_o = 0; n_o < (n_dim+mac_y-1)/mac_y; n_o++) {
		#pragma HLS LOOP_TRIPCOUNT min = n_size/mac_y max = n_size/mac_y
		loop_k:
			for (int k = 0; k < k_dim; k++) {
			#pragma HLS LOOP_TRIPCOUNT min = k_size max = k_size
			loop_m_i:
				for (int m_i = 0; m_i < mac_x; m_i++) {
				#pragma HLS UNROLL factor = mac_x
				loop_n_i:
					for (int n_i = 0; n_i < mac_y; n_i++) {
					#pragma HLS UNROLL factor = mac_y
						// Get previous sum

						int m = m_o*mac_x+m_i;
						int n = n_o*mac_y+n_i;

						int last = (k == 0) ? 0 : localO[m][n];

						// Update current sum
						// Handle boundary conditions
						int a_val = (m < m_dim && k < k_dim) ? localA[m][k] : 0;
						int b_val = (k < k_dim && n < n_dim) ? localB[k][n] : 0;
						int result = last + a_val * b_val;

						// Write back results
						localO[m][n] = result;
					}
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
