#!/usr/bin/env python3
"""Validate: did dextromethorphan's docked pose land near the heme Fe?"""
import numpy as np

FE = np.array([-12.796, 50.784, -6.164])

def pose_centroid(path, mode=1):
    atoms = []
    n_modes_seen = 0
    for line in open(path):
        if line.startswith("MODEL") or line.startswith("ROOT") and atoms:
            pass
        if line.startswith("ATOM"):
            atoms.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if line.startswith("ENDROOT"):
            n_modes_seen += 1
            if n_modes_seen >= mode:
                break
    a = np.array(atoms)
    return a.mean(axis=0), len(a)

# dm_out.pdbqt contains all 3 modes concatenated
c, n = pose_centroid("dm_out.pdbqt")
print(f"first pose: {n} atoms, centroid = ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
d = np.linalg.norm(c - FE)
print(f"distance to heme Fe = {d:.2f} A")
if d < 8:
    print("VALID: pose is in the heme active site (box is correct)")
else:
    print(f"WARNING: pose is {d:.1f}A from Fe - box may be off")
