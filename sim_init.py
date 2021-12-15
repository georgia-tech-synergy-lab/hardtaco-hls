# Create new simulation directory

import csv
import argparse
import sys
import os
import fileinput


# Read matrix workload configuration file 
def read_cfg_file(cfg_file):
	file1 = open(str(cfg_file), 'r')
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

# Read parameter file 
def read_param_file(param_file):
	file1 = open(param_file, 'r')
	Lines = file1.readlines()
	
	num_pes = -1
	tpu_pes_x = -1
	tpu_pes_y = -1
	storage_m_dim = -1
	storage_n_dim = -1
	storage_k_dim = -1
	storage_mk_nnz = -1
	storage_kn_nnz = -1
	workload = ""
 
	# parse tensor config line
	for line in Lines: 
		if (line.startswith('NUM_PES')):
			line_split = line.split(":")
			num_pes = int(line_split[1])
		elif (line.startswith('TPU_PES_X')):
			line_split = line.split(":")   
			tpu_pes_x = int(line_split[1])
		elif (line.startswith('TPU_PES_Y')):
			line_split = line.split(":")   
			tpu_pes_y = int(line_split[1])
		elif (line.startswith('STORAGE_M_DIM')):
			line_split = line.split(":")   
			storage_m_dim = int(line_split[1])
		elif (line.startswith('STORAGE_N_DIM')):
			line_split = line.split(":")   
			storage_n_dim = int(line_split[1])
		elif (line.startswith('STORAGE_K_DIM')):
			line_split = line.split(":")   
			storage_k_dim = int(line_split[1])
		elif (line.startswith('STORAGE_MK_NNZ')):
			line_split = line.split(":")   
			storage_mk_nnz = int(line_split[1])
		elif (line.startswith('STORAGE_KN_NNZ')):
			line_split = line.split(":")   
			storage_kn_nnz = int(line_split[1])
		elif (line.startswith('WORKLOAD')):
			line_split = line.split(":")
			workload = line_split[1]
	
	return num_pes, tpu_pes_x, tpu_pes_y, storage_m_dim, storage_n_dim, storage_k_dim, storage_mk_nnz, storage_kn_nnz, workload


def main():
	# Read input arguments 
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--param_file", help="Parameterization File")
	args = parser.parse_args()

	print( "param_file {}".format(
			args.param_file
			))

	num_pes, tpu_pes_x, tpu_pes_y, storage_m_dim, storage_n_dim, storage_k_dim, storage_mk_nnz, storage_kn_nnz, workload = read_param_file(args.param_file)

	config_path = "./workloads/matrix_cfg/" + str(workload)
	m_dim, n_dim, k_dim, mk_nnz, kn_nnz = read_cfg_file(config_path)

	# hardware storage parameter check
	print(num_pes, tpu_pes_x, tpu_pes_y, storage_m_dim, storage_n_dim, storage_k_dim, storage_mk_nnz, storage_kn_nnz, workload)
	assert m_dim <= storage_m_dim, "Storage size smaller than workload M dim, need to (1) increase storage or (2) tile workload"
	assert n_dim <= storage_n_dim, "Storage size smaller than workload N dim, need to (1) increase storage or (2) tile workload"
	assert k_dim <= storage_k_dim, "Storage size smaller than workload K dim, need to (1) increase storage or (2) tile workload"
	assert mk_nnz <= storage_mk_nnz, "Storage size smaller than MK nnzs, need to (1) increase storage or (2) tile workload"
	assert kn_nnz <= storage_kn_nnz, "Storage size smaller than KN nnzs, need to (1) increase storage or (2) tile workload"

	# run configuration generation script
	if (workload == "example.cfg"):
		os.system('python3 ./workloads/gen_testbench.py -cfg ./workloads/matrix_cfg/example.cfg -r random')
	elif (workload == "mesh1e1.cfg"):
		os.system('python3 ./workloads/gen_testbench.py -cfg ./workloads/matrix_cfg/mesh1e1.cfg -mtx ./workloads/suitesparse/mesh1e1/mesh1e1.mtx -u undirected -r real')
	elif (workload == "journals.cfg"):
		os.system('python3 ./workloads/gen_testbench.py -cfg ./workloads/matrix_cfg/journals.cfg -mtx ./workloads/suitesparse/Journals/Journals.mtx -u undirected -r real')
	elif (workload == "685_bus.cfg"):
		os.system('python3 ./workloads/gen_testbench.py -cfg ./workloads/matrix_cfg/685_bus.cfg -mtx ./workloads/suitesparse/685_bus/685_bus.mtx -u undirected -r real')

	# copy testbench file to respective location
	tb_path = ["./sim/cpp_kernels/tpu-like/data","./sim/cpp_kernels/eie-like/data", \
			"./sim/cpp_kernels/extensor-like/data", "./sim/cpp_kernels/outerspace-like/data", \
			"./sim/cpp_kernels/matraptor-like/data"]
	for i in tb_path:
		cmd_str = "cp output_O.csv " + str(i)
		os.system(cmd_str)
		cmd_str = "cp " + str(config_path) + " " + str(i) + "/input_cfg.csv"
		os.system(cmd_str)

	tb_path = ["./sim/cpp_kernels/tpu-like/data"]
	for i in tb_path:
		cmd_str = "cp input_A.csv " + str(i)
		os.system(cmd_str)

	tb_path = ["./sim/cpp_kernels/tpu-like/data","./sim/cpp_kernels/eie-like/data"]
	for i in tb_path:
		cmd_str = "cp input_B.csv " + str(i)
		os.system(cmd_str)

	tb_path = ["./sim/cpp_kernels/eie-like/data","./sim/cpp_kernels/extensor-like/data"]
	for i in tb_path:
		cmd_str = "cp input_A_csr*.csv " + str(i)
		os.system(cmd_str)

	tb_path = ["./sim/cpp_kernels/outerspace-like/data","./sim/cpp_kernels/matraptor-like/data"]
	for i in tb_path:
		cmd_str = "cp input_A_csc*.csv " + str(i)
		os.system(cmd_str)

	tb_path = ["./sim/cpp_kernels/extensor-like/data","./sim/cpp_kernels/matraptor-like/data"]
	for i in tb_path:
		cmd_str = "cp input_B_csc*.csv " + str(i)
		os.system(cmd_str)

	tb_path = ["./sim/cpp_kernels/outerspace-like/data"]
	for i in tb_path:
		cmd_str = "cp input_B_csr*.csv " + str(i)
		os.system(cmd_str)

	os.system('rm *csv')

	# change HLS cpp_kernel parameters 
	cpp_path = ["./sim/cpp_kernels/tpu-like/src/","./sim/cpp_kernels/eie-like/src/", \
			"./sim/cpp_kernels/extensor-like/src/", "./sim/cpp_kernels/outerspace-like/src/", \
			"./sim/cpp_kernels/matraptor-like/src/"]

	for i in cpp_path:
	
		filename_host = str(i) + "host.cpp"
		filename_mmult = str(i) + "mmult.cpp"
	
		for line in fileinput.input([filename_host], inplace=True):
			if line.strip().startswith('#define MATRIX_SIZE_M'):
				line = '#define MATRIX_SIZE_M ' + str(m_dim) + '\n'
			elif line.strip().startswith('#define MATRIX_SIZE_K'):
				line = '#define MATRIX_SIZE_K ' + str(k_dim) + '\n'
			elif line.strip().startswith('#define MATRIX_SIZE_N'):
				line = '#define MATRIX_SIZE_N ' + str(n_dim) + '\n'	
			elif line.strip().startswith('#define STORAGE_M_DIM'):
				line = '#define STORAGE_M_DIM ' + str(storage_m_dim) + '\n'
			elif line.strip().startswith('#define STORAGE_K_DIM'):
				line = '#define STORAGE_K_DIM ' + str(storage_k_dim) + '\n'	
			elif line.strip().startswith('#define STORAGE_N_DIM'):
				line = '#define STORAGE_N_DIM ' + str(storage_n_dim) + '\n'					
			elif line.strip().startswith('#define MK_NNZ'):
				line = '#define MK_NNZ ' + str(mk_nnz) + '\n'
			elif line.strip().startswith('#define KN_NNZ'):
				line = '#define KN_NNZ ' + str(kn_nnz) + '\n'	
			elif line.strip().startswith('#define STORAGE_MK_NNZ'):
				line = '#define STORAGE_MK_NNZ ' + str(storage_mk_nnz) + '\n'	
			elif line.strip().startswith('#define STORAGE_KN_NNZ'):
				line = '#define STORAGE_KN_NNZ ' + str(storage_kn_nnz) + '\n'				
			sys.stdout.write(line)
			
		for line in fileinput.input([filename_mmult], inplace=True):
			if line.strip().startswith('#define MATRIX_SIZE_M'):
				line = '#define MATRIX_SIZE_M ' + str(m_dim) + '\n'
			elif line.strip().startswith('#define MATRIX_SIZE_K'):
				line = '#define MATRIX_SIZE_K ' + str(k_dim) + '\n'
			elif line.strip().startswith('#define MATRIX_SIZE_N'):
				line = '#define MATRIX_SIZE_N ' + str(n_dim) + '\n'	
			elif line.strip().startswith('#define STORAGE_M_DIM'):
				line = '#define STORAGE_M_DIM ' + str(storage_m_dim) + '\n'
			elif line.strip().startswith('#define STORAGE_K_DIM'):
				line = '#define STORAGE_K_DIM ' + str(storage_k_dim) + '\n'	
			elif line.strip().startswith('#define STORAGE_N_DIM'):
				line = '#define STORAGE_N_DIM ' + str(storage_n_dim) + '\n'		
			elif line.strip().startswith('#define MK_NNZ'):
				line = '#define MK_NNZ ' + str(mk_nnz) + '\n'
			elif line.strip().startswith('#define KN_NNZ'):
				line = '#define KN_NNZ ' + str(kn_nnz) + '\n'
			elif line.strip().startswith('#define STORAGE_MK_NNZ'):
				line = '#define STORAGE_MK_NNZ ' + str(storage_mk_nnz) + '\n'	
			elif line.strip().startswith('#define STORAGE_KN_NNZ'):
				line = '#define STORAGE_KN_NNZ ' + str(storage_kn_nnz) + '\n'		
			elif line.strip().startswith('#define NUM_MACS'):
				line = '#define NUM_MACS ' + str(num_pes) + '\n'
			elif line.strip().startswith('#define NUM_MAC_X'):
				line = '#define NUM_MAC_X ' + str(tpu_pes_x) + '\n'	
			elif line.strip().startswith('#define NUM_MAC_Y'):
				line = '#define NUM_MAC_Y ' + str(tpu_pes_y) + '\n'					
			sys.stdout.write(line)

main()

