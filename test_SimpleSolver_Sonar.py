import numpy as np
from SimpleSolver import SimpleSolver
from getParam_Sonar import *
from eval_f_Sonar import *
from eval_u_Sonar import *
from VisualizeState import *
import pdb

print('\n**************** test_SimpleSolverHeatBar.py *************')
print('Running: test_SimpleSolverHeatBar.py')
print('tests 1D linear diffusion on a heat conducting bar')
print('showing time domain simulation as debugging tool\n')

# select input evaluation functions
eval_u = eval_u_Sonar

# Heat Conducting Bar Example
p, x_start, t_start, t_stop, w_max = getParam_Sonar(Nx,Nz,Lx,Lz,UseSparseMatrices=False)
eval_f = eval_f_Sonar

visualize = True
w = w_max
num_iter = np.ceil((t_stop - t_start) / w)

[X, t] = SimpleSolver(eval_f, x_start, p, eval_u, num_iter, w, visualize, gif_file_name="test_SimpleSolver_Sonar.gif")
