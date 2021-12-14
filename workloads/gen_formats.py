#############################################################################################
# Compression Format helper functions for testing sub-accelerators
#############################################################################################

import math
import numpy as np
import csv
import fileinput


# ---------------------------------------------------------
# generate random matrix given dimension and nnz
# ---------------------------------------------------------
def gen_random_matrix(m_dim, k_dim, mk_nnz):
	zeros = np.zeros(m_dim*k_dim-mk_nnz, dtype=int)
	rand = np.random.randint(1,10,(mk_nnz), dtype=int)
	arr=np.concatenate((rand,zeros), axis=0, out=None)
	np.random.shuffle(arr)
	matrix = np.reshape(arr, (m_dim, k_dim))
	return matrix

# ---------------------------------------------------------
# convert dense matrix to CSR/CSC format
# ---------------------------------------------------------
def gen_dense2csx(matrix, mode):
	values = []
	m_dim = np.shape(matrix)[0]
	k_dim = np.shape(matrix)[1]
	if (mode == "csr"):
		ptr = [0] * (np.shape(matrix)[0] + 1)
	else:
		ptr = [0] * (np.shape(matrix)[1] + 1)
	idx = []

	# mode support dense2csr, or dense2csc
	if (mode != "csr" and mode != "csc"):
		raise Exception("Pick mode {csr, csc}")

	if (mode == "csc"):
		matrix = np.transpose(matrix)

	# iterate and fill in nnz, ptr, idx
	for x_idx, x_vec in enumerate(matrix):
		for y_idx, val in enumerate(x_vec):
			if (val != 0):
				values.append(val)
				ptr[x_idx+1] = ptr[x_idx+1]+1
				idx.append(y_idx)

	# conduct prefix sum on ptr
	for ptr_idx, val in enumerate(ptr):
		if (ptr_idx != 0):
			ptr[ptr_idx] += ptr[ptr_idx-1]

	# return compression format dictionary
	csx_dict = {
		"values": values,
		"idx": idx,
		"ptr": ptr,
		"mode": mode,
		"m_dim": m_dim,
		"k_dim": k_dim
	}
	return csx_dict

# ---------------------------------------------------------
# convert CSR to CSC format (and vice versa)
# ---------------------------------------------------------
def gen_csx2csx(csx_dict, target_dim):
	i_value = csx_dict['values']
	i_col_id = csx_dict['idx']
	i_row_ptr = csx_dict['ptr']
	i_m_dim = csx_dict['m_dim']
	i_k_dim = csx_dict['k_dim']

	nnz = len(i_value)

	# init new format structures
	o_value = [0] * nnz
	o_row_id = [0] * nnz
	o_col_ptr = [0] * (target_dim + 1) 
	
	# build col ptr values
	for idx in i_col_id:
		o_col_ptr[int(idx)+1] = o_col_ptr[int(idx)+1] + 1

	# prefix sum on o_col_ptr
	for idx, val in enumerate(o_col_ptr):
		if (idx != 0): 
			o_col_ptr[idx] += o_col_ptr[idx-1]
	
	col_ptr_copy = o_col_ptr.copy()

	# update value and row_id values
	count = 0
	row_ptr_pos = 1
	for value in i_value:
		#print("value", value)
		# find row_id and update o_row_id
		if (i_row_ptr[row_ptr_pos] > count):
			#print("in current line, increase cnt")
			o_row_id[col_ptr_copy[i_col_id[count]]] = int(row_ptr_pos - 1)
		else:
			#print("in next line, increase row_ptr_pos")
			for i in range(len(i_row_ptr)-row_ptr_pos-1):
				if (i_row_ptr[row_ptr_pos] == i_row_ptr[row_ptr_pos+1]):
					row_ptr_pos = row_ptr_pos + 1
				else:
					row_ptr_pos = row_ptr_pos + 1
					break

		o_row_id[col_ptr_copy[i_col_id[count]]] = int(row_ptr_pos - 1)
		#print("row_ptr_pos placed", row_ptr_pos - 1)

		# update ouputs
		o_value[col_ptr_copy[i_col_id[count]]] = int(value)
		
		# in place add col_ptr_copy for indexing
		col_ptr_copy[i_col_id[count]] = col_ptr_copy[i_col_id[count]] + 1
		
		# iterate to next
		count = count + 1

	# update mode field
	mode = "none defined"
	if (csx_dict['mode'] == "csc"):
		mode = "csr"
	elif (csx_dict['mode'] == "csr"):
		mode = "csc"

	# return compression format dictionary
	new_csx_dict = {
		"values": o_value,
		"idx": o_row_id,
		"ptr": o_col_ptr,
		"mode": mode,
		"m_dim": i_m_dim,
		"k_dim": i_k_dim
	}
	return new_csx_dict

# ---------------------------------------------------------
# convert CSR/CSC format to dense (uncompressed)
# ---------------------------------------------------------
def gen_csx2dense(csx_dict):
	i_val = csx_dict['values']
	i_idx = csx_dict['idx']
	i_ptr = csx_dict['ptr']
	i_mode = csx_dict['mode']
	m_dim = csx_dict['m_dim']
	k_dim = csx_dict['k_dim']

	dense_array = np.zeros([m_dim, k_dim], dtype=int)

	count = 0

	for ptr_pos, ptr_val in enumerate(i_ptr):
		# skip the first value of the ptr field
		if (ptr_pos != 0):
			while (ptr_val > count):
				if (i_mode == "csr"):
					dense_array[ptr_pos-1][i_idx[count]] = int(i_val[count])
				elif (i_mode == "csc"):
					dense_array[i_idx[count]][ptr_pos-1] = int(i_val[count])
				count = count + 1

	return dense_array

# ---------------------------------------------------------
# convert uncompressed to Bitmask format
# ---------------------------------------------------------
def gen_dense2bitmask(matrix, mode):

	values = []
	bitmask = []

	m_dim = np.shape(matrix)[0]
	k_dim = np.shape(matrix)[1]

	if (mode == "row"):
		for x_idx, x_vec in enumerate(matrix):
			for y_idx, val in enumerate(x_vec):
				if (val != 0):
					values.append(val)
					bitmask.append(1)
				else:
					bitmask.append(0)
	elif (mode == "col"):
		matrix = np.transpose(matrix)
		for x_idx, x_vec in enumerate(matrix):
			for y_idx, val in enumerate(x_vec):
				if (val != 0):
					values.append(val)
					bitmask.append(1)
				else:
					bitmask.append(0)

	# return compression format dictionary
	bitmask_dict = {
		"values": values,
		"bitmask": bitmask,
		"m_dim": m_dim,
		"k_dim": k_dim,
		"mode": mode
	}
	return bitmask_dict

# ---------------------------------------------------------
# convert Bitmask to uncompressed
# ---------------------------------------------------------
def gen_bitmask2dense(bitmask_dict):
	values = bitmask_dict['values']
	bitmask = bitmask_dict['bitmask']
	m_dim = bitmask_dict['m_dim']
	k_dim = bitmask_dict['k_dim']
	mode = bitmask_dict['mode']

	dense_array = np.zeros([m_dim, k_dim], dtype=int)

	if (mode == "row"):
		for id, bit in enumerate(bitmask):
			if (bit == 1):
				m_id = int(id) / int(k_dim)
				k_id = int(id) % int(k_dim)
				dense_array[int(m_id)][int(k_id)] = values[0]
				values.pop(0)
	elif (mode == "col"):
		for id, bit in enumerate(bitmask):
			if (bit == 1):
				m_id = int(id) / int(m_dim)
				k_id = int(id) % int(m_dim)
				dense_array[int(k_id)][int(m_id)] = values[0]
				values.pop(0)

	return dense_array

# ---------------------------------------------------------
# convert uncompressed to RLC format
# ---------------------------------------------------------
def gen_dense2rlc(matrix, mode, runs):

	rlc = []
	m_dim = np.shape(matrix)[0]
	k_dim = np.shape(matrix)[1]

	run_cnt = 0
	if (mode == "row"):
		for x_idx, x_vec in enumerate(matrix):
			for y_idx, val in enumerate(x_vec):
				if (val != 0):
					rlc.append(run_cnt)
					rlc.append(val)
					run_cnt = 0
				elif (run_cnt == runs):
					rlc.append(runs)
					rlc.append(0)
					run_cnt = 0
				else:
					run_cnt += 1	
	elif (mode == "col"):
		matrix = np.transpose(matrix)
		for x_idx, x_vec in enumerate(matrix):
			for y_idx, val in enumerate(x_vec):
				if (val != 0):
					rlc.append(run_cnt)
					rlc.append(val)
					run_cnt = 0
				elif (run_cnt == runs):
					rlc.append(runs)
					rlc.append(0)
					run_cnt = 0
				else:
					run_cnt += 1

	# return compression format dictionary
	rlc_dict = {
		"rlc": rlc,
		"m_dim": m_dim,
		"k_dim": k_dim,
		"mode": mode
	}
	return rlc_dict

# ---------------------------------------------------------
# convert RLC to uncompressed
# ---------------------------------------------------------
def gen_rlc2dense(rlc_dict):
	rlc = rlc_dict['rlc']
	m_dim = rlc_dict['m_dim']
	k_dim = rlc_dict['k_dim']
	mode = rlc_dict['mode']

	dense_array = np.zeros([m_dim, k_dim], dtype=int)

	run_cnt = 0
	if (mode == "row"):
		for id, field in enumerate(rlc):
			if (id % 2 == 0): # runs
				run_cnt += field
			else: # values
				m_id = int(run_cnt) / int(k_dim)
				k_id = int(run_cnt) % int(k_dim)
				dense_array[int(m_id)][int(k_id)] = field
				run_cnt += 1

	elif (mode == "col"):
		for id, field in enumerate(rlc):
			if (id % 2 == 0): # runs
				run_cnt += field
			else: # values
				m_id = int(run_cnt) / int(m_dim)
				k_id = int(run_cnt) % int(m_dim)
				dense_array[int(k_id)][int(m_id)] =	field
				run_cnt += 1

	return dense_array

# ---------------------------------------------------------
# convert Dense to COO
# ---------------------------------------------------------
def gen_dense2coo(matrix):

	values = []
	m_list = []
	k_list = []
	m_dim = np.shape(matrix)[0]
	k_dim = np.shape(matrix)[1]

	# iterate and fill in nnz, ptr, idx
	for x_idx, x_vec in enumerate(matrix):
		for y_idx, val in enumerate(x_vec):
			if (val != 0):
				values.append(val)
				m_list.append(x_idx)
				k_list.append(y_idx)

	# return compression format dictionary
	coo_dict = {
		"values": values,
		"m_list": m_list,
		"k_list": k_list,
		"m_dim": m_dim,
		"k_dim": k_dim
	}
	return coo_dict

# ---------------------------------------------------------
# convert COO to Dense
# ---------------------------------------------------------
def gen_coo2dense(coo_dict):
	values = coo_dict['values']
	m_list = coo_dict['m_list']
	k_list = coo_dict['k_list']
	m_dim = coo_dict['m_dim']
	k_dim = coo_dict['k_dim']

	dense_array = np.zeros([m_dim, k_dim], dtype=int)

	for i in range(len(values)):
		dense_array[m_list[i]][k_list[i]] = values[i]

	return dense_array

# ---------------------------------------------------------
# convert COO to CSX 
# useful for hyper+ sparsity & natural compression
# ---------------------------------------------------------
def gen_coo2csx(coo_dict, mode):

	values = coo_dict['values']
	m_list = coo_dict['m_list']
	k_list = coo_dict['k_list']
	m_dim = coo_dict['m_dim']
	k_dim = coo_dict['k_dim']

	o_values = [0] * len(values)
	o_id = [0] * len(values)
	o_ptr = []

	if (mode == "csr"):
		o_ptr = [0] * (m_dim+1)

		# build ptr fields
		for i, m in enumerate(m_list):
			o_ptr[m+1] += 1

		# do prefix sum
		for idx, val in enumerate(o_ptr):
			if (idx != 0): 
				o_ptr[idx] += o_ptr[idx-1]

		# fill values and ids
		ptr_copy = o_ptr.copy()
		for i, val in enumerate(values):
			o_values[ptr_copy[m_list[i]]] = val
			o_id[ptr_copy[m_list[i]]] = k_list[i]
			ptr_copy[m_list[i]] += 1
	else:
		o_ptr = [0] * (k_dim+1)

		# build ptr fields
		for i, k in enumerate(k_list):
			o_ptr[k+1] += 1

		# do prefix sum
		for idx, val in enumerate(o_ptr):
			if (idx != 0): 
				o_ptr[idx] += o_ptr[idx-1]

		# fill values and ids
		ptr_copy = o_ptr.copy()
		for i, val in enumerate(values):
			o_values[ptr_copy[k_list[i]]] = val
			o_id[ptr_copy[k_list[i]]] = m_list[i]
			ptr_copy[k_list[i]] += 1
			
	# return compression format dictionary
	csx_dict = {
		"values": o_values,
		"idx": o_id,
		"ptr": o_ptr,
		"mode": mode,
		"m_dim": m_dim,
		"k_dim": k_dim
	}
	return csx_dict


#############################################################################################
# Test cases for helper functions
#############################################################################################
def test_converter():
	m_dim = 50
	k_dim = 12
	nnz = 42
	matrix = gen_random_matrix(m_dim,k_dim,nnz)
	print(matrix)
	csr = gen_dense2csx(matrix, "csr")
	print(csr)
	csc = gen_dense2csx(matrix, "csc")
	print(csc)
	csc_conv = gen_csx2csx(csr, k_dim)
	if not (csc_conv == csc):
		raise Exception("FAIL")
	dense_from_csr = gen_csx2dense(csr)
	if not (np.array_equal(dense_from_csr, matrix)):
		raise Exception("FAIL")
	dense_from_csc = gen_csx2dense(csc_conv)
	if not (np.array_equal(dense_from_csc, matrix)):
		raise Exception("FAIL")
	bitmask_row = gen_dense2bitmask(matrix, "row")
	bitmask_col = gen_dense2bitmask(matrix, "col")
	dense_bit_row = gen_bitmask2dense(bitmask_row)
	if not (np.array_equal(dense_bit_row, matrix)):
		raise Exception("FAIL")
	dense_bit_col = gen_bitmask2dense(bitmask_col)
	if not (np.array_equal(dense_bit_col, matrix)):
		raise Exception("FAIL")
	rlc_row = gen_dense2rlc(matrix, "row", 4)
	rlc_col = gen_dense2rlc(matrix, "col", 4)
	dense_rlc_row = gen_rlc2dense(rlc_row)
	if not (np.array_equal(dense_rlc_row, matrix)):
		raise Exception("FAIL")
	dense_rlc_col = gen_rlc2dense(rlc_col)
	if not (np.array_equal(dense_rlc_col, matrix)):
		raise Exception("FAIL")
	coo = gen_dense2coo(matrix)
	dense_coo = gen_coo2dense(coo)
	if not (np.array_equal(dense_coo, matrix)):
		raise Exception("FAIL")
	csrfromcoo = gen_coo2csx(coo, "csr")
	if not (csrfromcoo == csr):
		raise Exception("FAIL")
	cscfromcoo = gen_coo2csx(coo, "csc")
	if not (cscfromcoo == csc):
		raise Exception("FAIL")
