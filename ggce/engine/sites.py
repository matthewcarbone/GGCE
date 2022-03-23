import numpy as np


M = 4
N = 3


def delta(x, y):
	if x == y:
		return 1
	if x != y:
		return 0


def one_dimension():
	pops = [[]]*(N + 1)
	for i in range(N + 1):
		if i == 0:
			pops[0] = [0]
		elif i == 1:
			pops[1] = [1]
		elif i == 2:
			pops_n = []*M
			pops_n[0] = [2]
			for j in range(2, M+1):
				new_pop = [0]*j
				new_pop[0] = 1
				new_pop[len(new_pop) - 1] = 1
				pops_n[j-1] = new_pop
				pops[i] = pops_n
		if i > 2:
			new_pops = []
			for pop1 in pops[i-1]:
				for k in range(len(pop1)):
					pop2 = [n + delta(index, k) for index, n in enumerate(pop1)]
					if pop2 not in new_pops:
						new_pops.append(pop2)
			pops[i] = new_pops


def two_dimensions():
	# the nth element of pops is a list of arrays with n bosons
	pops = [[]]*(N + 1)
	for i in range(N + 1):
		# manually make the 0 and 1 boson arrays
		if i == 0:
			pops[0] = np.array([0])
		elif i == 1:
			pops[1] = np.array([1])
		elif i == 2:
			# make the 4 allowed 2 boson arrays of the M sizes
			new_pops = [[2]]
			for j in range(2, M+1):
				coords = [[0, j-1], [j-1, 0], [j-1, j-1]]
				for coord in coords:
					new_pop = np.zeros([j, j], dtype=int)
					new_pop[0][0] = 1
					new_pop[coord[0]][coord[1]] = 1
					new_pops.append(np.array(new_pop))
				pops[i] = new_pops
		elif i > 2:
			# go through the list of i-1 boson arrays, and add a boson to each site one by one
			# append if the new array i-boson array isn't already in new_pops
			new_pops = []
			for pop1 in pops[i-1]:
				if np.array_equal(pop1, np.array([i-1])):
					new_pops.append(np.array([i]))
				else:
					lens = np.shape(pop1)
					for k in range(lens[0]):
						for m in range(lens[1]):
							plus_one = np.zeros([lens[0], lens[1]])
							plus_one[k, m] = 1
							pop2 = pop1 + plus_one

							check = np.array(new_pops)
							will_append = True
							for test in check:
								if np.array_equal(test, pop2):
									will_append = False
							if will_append:
								new_pops.append(pop2)

			pops[i] = new_pops

	return pops


# do the reduction rules for annihilation operators
# need a method to shift around array slices for illegal arrays created by annihilation
def shift(pop, side):
	shape = np.shape(pop)
	length = shape[0]
	new_pop = np.zeros(shape)
	if side == 'left':
		new_pop[:, 0:length - 1] = pop[:, 1:length]
	if side == 'top':
		new_pop[0:length - 1, :] = pop[1:length, :]
	return new_pop


# this method reshuffles the arrays and returns the legal array with the number of x and y shifts it took
def shuffle(pop, xshift, yshift):
	length = np.shape(pop)[0]
	zeros = np.zeros(length, dtype=int)
	shifted = False

	# check to see if we need to shift things into the corner
	# if yes, increment the phase factors by 1
	if np.array_equal(pop[0, :], zeros):
		pop = shift(pop, 'top')
		yshift += 1
		shifted = True
	if np.array_equal(pop[:, 0], zeros):
		pop = shift(pop, 'left')
		xshift += 1
		shifted = True
	# if the array has 0s that are artificially increasing its size, chop them off
	if np.array_equal(pop[length - 1, :], zeros) and np.array_equal(pop[:, length - 1], zeros):
		pop = pop[0:length - 1, 0:length - 1]
		shifted = True

	if shifted:
		return shuffle(pop, xshift, yshift)
	if not shifted:
		return pop, xshift, yshift


def annihilate2(pop, x, y):
	coefficient = pop[x, y]
	if pop[x, y] > 0:
		pop[x, y] -= 1
	pop, xshift, yshift = shuffle(pop, 0, 0)

	return coefficient, pop, xshift, yshift


def creations(pop):
	new_creations = []
	shape = np.shape(pop)
	length = shape[0]
	delta = M - length
	for i in range(length):
		for j in range(length):
			plus_one = np.zeros(shape)
			plus_one[i, j] = 1
			new_creations.append([pop + plus_one, 0, 0])

	for m in range(delta):
		expanded_pop = np.zeros((length + m, length + m))
		expanded_pop[m:m + length, m:m + length] = pop
		if m == 0:
			plus_one = np.zeros((length + m, length + m))
			plus_one[0, 0] = 1
			new_pop, x, y = shuffle(plus_one + pop, 0, 0)
			new_creations.append([new_pop, m, m])
		for i in range(1, length + m):
			plus_one = np.zeros((length + m, length + m))
			plus_one[0, i] = 1
			new_pop, x, y = shuffle(plus_one + expanded_pop, 0, 0)
			new_creations.append([new_pop, 0, m])
		for i in range(1, length + m):
			plus_one = np.zeros((length + m, length + m))
			plus_one[i, 0] = 1
			new_pop, x, y = shuffle(plus_one + expanded_pop, 0, 0)
			new_creations.append([new_pop, m, 0])
		for i in range(length + m):
			plus_one = np.zeros((length + m, length + m))
			plus_one[length + m - 1, i] = 1
			new_pop, x, y = shuffle(plus_one + expanded_pop, 0, 0)
			new_creations.append([new_pop, 0, 0])
		for i in range(length + m):
			plus_one = np.zeros((length + m, length + m))
			plus_one[i, length + m - 1] = 1
			new_pop, x, y = shuffle(plus_one + expanded_pop, 0, 0)
			new_creations.append([new_pop, 0, 0])

	return new_creations


test_pop = np.array([[1, 0], [0, 1]])
# print(shuffle(test_pop, , 0))
# print(annihilate2(test_pop, 0, 0))
