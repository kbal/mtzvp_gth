import numpy as np
from ase import io
from ase.calculators.cp2k import CP2K
from ase.optimize import BFGS
from ase.units import kcal,mol,Rydberg

# set up CP2K
calc = CP2K(label='calc/cp2k',
            command='mpiexec -np 4 cp2k_shell.popt',
            charge=0,
            cutoff=800 * Rydberg,
            xc='',
            basis_set='mTZVP-GTH',
            pseudo_potential='GTH-PBE',
            basis_set_file='BASIS_MTZVP',
            potential_file='GTH_POTENTIALS',
            inp='''&FORCE_EVAL
                     &DFT
                       &QS
                         METHOD GPW
                       &END QS
                       &SCF
                         &OT
                           PRECONDITIONER FULL_ALL
                           MINIMIZER DIIS
                         &END OT
                       &END SCF
                       &XC
                         &XC_FUNCTIONAL
                           &BECKE97
                             SCALE_X 1.0
                             SCALE_C 1.0
                             PARAMETRIZATION B97-3c
                           &END BECKE97
                         &END XC_FUNCTIONAL
                         &vdW_POTENTIAL
                           DISPERSION_FUNCTIONAL PAIR_POTENTIAL
                           &PAIR_POTENTIAL
                             TYPE DFTD3(BJ)
                             PARAMETER_FILE_NAME dftd3.dat
                             REFERENCE_FUNCTIONAL B97-3c
                             R_CUTOFF 7.93766
                             CALCULATE_C9_TERM
                             SHORT_RANGE_CORRECTION
                           &END PAIR_POTENTIAL
                         &END vdW_POTENTIAL
                       &END XC
                     &END DFT
                   &END FORCE_EVAL
                   ''')

# periodic system
cryst = io.read('cryst.cif')
ncryst = len(cryst.get_atomic_numbers())

# single molecule
molec = io.read('mol.xyz')
molec.set_cell(10 * np.identity(3))
nmol = len(molec.get_atomic_numbers())

# size of unit cell
size = ncryst/nmol

# run calcs
cryst.set_calculator(calc)
dyn = BFGS(cryst, trajectory='opt_cryst.traj')
dyn.run(fmax=0.05)
e_cryst = cryst.get_potential_energy()

molec.set_calculator(calc)
dyn = BFGS(molec, trajectory='opt_mol.traj')
dyn.run(fmax=0.05)
e_mol = molec.get_potential_energy()

e_coh = (e_cryst - size*e_mol)/size
print("Cohesive energy (kcal/mol): ", e_coh/(kcal/mol))

