import ase.io
import numpy as np
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("structures_path", help="Path to an xyz file with structures", type = str)
parser.add_argument("r_cut", help = "cutoff radius", type = float)
parser.add_argument("--num_strucs_use", help="Number of random structures to use to estimate statics. If not specified, all the provided structures are used", type = str)

args = parser.parse_args()

structures_path = args.structures_path

structures = ase.io.read(structures_path, index = ':')

if args.num_strucs_use is not None:
    permutation = np.random.permutation(len(structures))
    structures = [structures[index] for index in permutation[:args.num_strucs_use]]

atom_nums = [len(struc.positions) for struc in structures]
total_num = np.sum(atom_nums)


p = Pool(cpu_count())


def get_n_neighbors(structure):
    i_list, j_list = ase.neighborlist.neighbor_list('ij', structure, args.r_cut)
    return len(i_list)


num_neighbors = p.map(get_n_neighbors, structures)
total_neighbors = np.sum(num_neighbors)

average_n_neighbors = total_neighbors / total_num

print(f"average number of neighbors within r_cut = {args.r_cut} is {average_n_neighbors}")
print("total number of atomic environment used to compute statistics is :", total_num)