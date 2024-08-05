import ase.io
import numpy as np


structures = ase.io.read("ethanol_reduced_100.xyz", ":")
np.random.shuffle(structures)
train = structures[:50]
valid = structures[50:60]
test = structures[60:]

ase.io.write("train.xyz", train)
ase.io.write("valid.xyz", valid)
ase.io.write("test.xyz", test)
