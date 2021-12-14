#######################################################################################
# Generate random/real sparse matrix given configuration file 
# and calculate golden output

# MTX File is only needed to generate real data

# Example execution: 
# python3 gen_testbench.py -cfg matrix_cfg/example.cfg -r random
# python3 gen_testbench.py -cfg matrix_cfg/journals.cfg -mtx suitesparse/Journals/Journals.mtx -u undirected -r real
# python3 gen_testbench.py -cfg matrix_cfg/685_bus.cfg -mtx suitesparse/685_bus/685_bus.mtx -u undirected -r real
# python3 gen_testbench.py -cfg matrix_cfg/mesh1e1.cfg -mtx suitesparse/mesh1e1/mesh1e1.mtx -u undirected -r real
#######################################################################################
import csv
import numpy as np
import math
from gen_formats import *
import argparse

# Read input arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--cfg_file", help="Matrix Configuration File")
parser.add_argument("-mtx", "--mtx_file", help="Matrix Market File")
parser.add_argument("-u", "--undirected", help="Mtx File either 'undirected' or 'directed'")
parser.add_argument("-r", "--random_or_real", help="Either 'random' or 'real'")
args = parser.parse_args()

print( "cfg_file {} mtx_file {} undirected {} random_or_real {} ".format(
		args.cfg_file,
		args.mtx_file,
		args.undirected,
		args.random_or_real
		))

		
def save_remove_last_char(file, model, mode):

	NEWLINE_SIZE_IN_BYTES = -1  # -2 on Windows?

	with open(file, 'wb') as fout:  # Note 'wb' instead of 'w'
		if (mode == 'c'):
			np.savetxt(fout, model, fmt='%i', newline=",")
			fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
			fout.truncate()
		elif (mode == 'u'):
			np.savetxt(fout, model.astype(int), fmt='%i', delimiter=",")
			fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
			fout.truncate()		
		
# ------------------------------------------------------------------------------------
# Helper Function to save compressed format to CSV files
# ------------------------------------------------------------------------------------
def save_compressed(csx_dict, matrix_name):
	val = csx_dict['values']
	idx = csx_dict['idx']
	ptr = csx_dict['ptr']
	mode = csx_dict['mode']
	m_dim = csx_dict['m_dim']
	k_dim = csx_dict['k_dim']
	
	val = [int(item) for item in val]
	
	val_str = "input_" + str(matrix_name) + "_" + str(mode) + "_val.csv"
	idx_str = "input_" + str(matrix_name) + "_" + str(mode) + "_idx.csv"
	ptr_str = "input_" + str(matrix_name) + "_" + str(mode) + "_ptr.csv"
	
	#np.savetxt(val_str, val, fmt='%i', newline=",")
	#np.savetxt(idx_str, idx, fmt='%i', newline=",")
	#np.savetxt(ptr_str, ptr, fmt='%i', newline=",")
	
	save_remove_last_char(val_str, val, 'c')
	save_remove_last_char(idx_str, idx, 'c')
	save_remove_last_char(ptr_str, ptr, 'c')

# ------------------------------------------------------------------------------------
# Helper Function to read configuration file
# ------------------------------------------------------------------------------------
def read_cfg_file():
	file1 = open(str(args.cfg_file), 'r')
	Lines = file1.readlines()
	
	m_dim = -1
	n_dim = -1
	k_dim = -1
	mk_nnz = -1
	kn_nnz = -1
 
	# parse tensor config line
	for line in Lines:
		if (not line.startswith('//')):
			print(line)
			line_split = line.split(",")

			print(line_split)
			m_dim = int(line_split[0])
			n_dim = int(line_split[1])
			k_dim = int(line_split[2])
			mk_nnz = int(line_split[3])
			kn_nnz = int(line_split[4])
	
	return m_dim, n_dim, k_dim, mk_nnz, kn_nnz

		
# ------------------------------------------------------------------------------------
# Generate random testbench values
# ------------------------------------------------------------------------------------
def gen_random_tb():

	# Get tensor configuration file data
	m_dim, n_dim, k_dim, mk_nnz, kn_nnz = read_cfg_file()
  
	# Generate Matrix A files (uncompressed, csr, csc)
	matrixA = gen_random_matrix(m_dim, k_dim, mk_nnz)
	#np.savetxt("input_A.csv", matrixA.astype(int), fmt='%i', delimiter=",")
	save_remove_last_char("input_A.csv", matrixA, 'u')
	matrixA_csr = gen_dense2csx(matrixA, "csr")
	save_compressed(matrixA_csr, "A")
	matrixA_csc = gen_dense2csx(matrixA, "csc")
	save_compressed(matrixA_csc, "A")

	# Generate Matrix B files (uncompressed, csr, csc)
	matrixB = gen_random_matrix(k_dim, n_dim, kn_nnz)
	#np.savetxt("input_B.csv", matrixB.astype(int), fmt='%i', delimiter=",")
	save_remove_last_char("input_B.csv", matrixB, 'u')
	matrixB_csr = gen_dense2csx(matrixB, "csr")
	save_compressed(matrixB_csr, "B")
	matrixB_csc = gen_dense2csx(matrixB, "csc")
	save_compressed(matrixB_csc, "B")
	
	# Generate Golden Output Matrix (uncompressed only)
	matrixO = np.matmul(matrixA, matrixB)
	#np.savetxt("output_O.csv", matrixO.astype(int), fmt='%i', delimiter=",")
	save_remove_last_char("output_O.csv", matrixO, 'u')

# ------------------------------------------------------------------------------------	
# Generate real data testbench values from suitespace (TODO)
# ------------------------------------------------------------------------------------
def gen_real_tb():

	# Get market matrix to COO format
	file1 = open(str(args.mtx_file), 'r')
	Lines = file1.readlines()
	
	start_flag = 1
	values_list = []
	m_list = []
	k_list = []
	m_dim = -1
	k_dim = -1
	
	# Iterate every line of MTX file
	for line in Lines:
		if (not line.startswith('%')):
		
			line_split = line.split( )
			
			# parse matrix market characteristic (first line)
			if (start_flag == 1):
				m_dim = int(line_split[0])
				k_dim = int(line_split[1])
				lines = int(line_split[2])
				start_flag = 0
			# append each coordinate value
			else: 
				m_idx = int(float(line_split[0]))-1 # minus one offset
				k_idx = int(float(line_split[1]))-1 # minus one offset
				value = int(math.ceil(float(line_split[2])))
				
				m_list.append(m_idx)
				k_list.append(k_idx)
				values_list.append(value)
				
				if (args.undirected == "undirected" and m_idx != k_idx):
					m_list.append(k_idx)
					k_list.append(m_idx)
					values_list.append(value)
								
				
	coo_dict = {
		"values": values_list,
		"m_list": m_list,
		"k_list": k_list,
		"m_dim": m_dim,
		"k_dim": k_dim
	}

	
	# Convert COO to Dense/CSR/CSC and save (for matrix A)
	matrixA = gen_coo2dense(coo_dict)
	#np.savetxt("input_A.csv", matrixA.astype(int), fmt='%i', delimiter=",")
	save_remove_last_char("input_A.csv", matrixA, 'u')
	matrixA_csr = gen_coo2csx(coo_dict, "csr")
	save_compressed(matrixA_csr, "A")
	matrixA_csc = gen_coo2csx(coo_dict, "csc")
	save_compressed(matrixA_csc, "A")
	
	# Get tensor configuration file data
	m_dim, n_dim, k_dim, mk_nnz, kn_nnz = read_cfg_file()
			
	# Generate Matrix B files (uncompressed, csr, csc)
	matrixB = gen_random_matrix(k_dim, n_dim, kn_nnz)
	#np.savetxt("input_B.csv", matrixB.astype(int), fmt='%i', delimiter=",")
	save_remove_last_char("input_B.csv", matrixB, 'u')
	matrixB_csr = gen_dense2csx(matrixB, "csr")
	save_compressed(matrixB_csr, "B")
	matrixB_csc = gen_dense2csx(matrixB, "csc")
	save_compressed(matrixB_csc, "B")
	
	# Generate Golden Output Matrix (uncompressed only)
	matrixO = np.matmul(matrixA, matrixB)
	#np.savetxt("output_O.csv", matrixO.astype(int), fmt='%i', delimiter=",")
	save_remove_last_char("output_O.csv", matrixO, 'u')
	

# ------------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------------	
def main():
	if (args.random_or_real == "real"):
		gen_real_tb()
	else:
		gen_random_tb()
		
		
main()
