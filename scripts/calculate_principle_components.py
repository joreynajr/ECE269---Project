"""
library(reticulate)
reticulate::repl_python()

Compute the principal components (PCs) using first 190 individuals’ neutral expression
image. Plot the singular values of the data matrix and justify your choice of
principal components.

(b) Reconstruct one of 190 individuals’ neutral expression image using different number
of PCs. As you vary the number of PCs, plot the mean squared error (MSE) of
reconstruction versus the number of principal components to show the accuracy of
reconstruction. Comment on your result.

(c) Reconstruct one of 190 individuals’ smiling expression image using different number
of PCs. Again, plot the MSE of reconstruction versus the number of principal
components and comment on your result.

(d) Reconstruct one of the other 10 individuals’ neutral expression image using different
number of PCs. Again, plot the MSE of reconstruction versus the number of principal
components and comment on your result.

(e) Use any other non-human image (e.g., car image, resize and crop to the same size),
and try to reconstruct it using all the PCs. Comment on your results.

(f) Rotate one of 190 individuals’ neutral expression image with different degrees and
try to reconstruct it using all PCs. Comment on your results.
"""

import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib import image
import glob 

def jpegs_to_matrix(jpegs):
  """
  converts a list of jpegs of the same size into a matrix.
  """
  data = []
  for fn in jpegs:
    img_mat = image.imread(fn).flatten()
    data.append(img_mat)
  data = np.mat(data)
  return(data)

# Loading all the datasets  
neutral_fns = 'data/frontalimages_spatiallynormalized_cropped_equalized_part*/*a.jpg'
neutral_fns = glob.glob(neutral_fns)
neutral_data = jpegs_to_matrix(neutral_fns)
neutral_190 = neutral_data[0:190]
neutral_single = neutral_data[190]

smiling_fns = 'data/frontalimages_spatiallynormalized_cropped_equalized_part*/*b.jpg'
smiling_fns = glob.glob(smiling_fns)
smiling_data = jpegs_to_matrix(smiling_fns)
smiling_single = smiling_data[190]

guitar = image.imread('figures/guitar_bw_196_162.jpg').flatten()

# Part A: calculate the principle components 
# - computer the PC's 
# - plot the singular values of the data matrix 

# Part B: reconstruct a neutral face using an increasing amount of PC's
# - reconstruct
# - plot the mean squared error vs PC's

# Part C: reconstruct a smiling face using an increasing amount of PC's
# - reconstruct
# - plot the mean squared error vs PC's

# Part D: reconstruct a neutral face (from the 10 unused samples) 
# using an increasing amount of PC's
# - reconstruct
# - plot the mean squared error vs PC's

# Part E: use any other non-human image 

# Part F: rotate an individuals neutral face and reconstruct with all PC's





