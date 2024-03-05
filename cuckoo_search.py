# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma as gamma_function
import math
import time
# import main class and related functions from the other file.
from cuckoo_functions import *

  
C_KOO = Cuckoo(Node_Density=300, Anchor_Ratio=0.35, R=25)

# Test Field Plot
C_KOO.plot_graph()

# Main Funcation getting called
start = time.time()
C_KOO.main()
stop = time.time()

# METRICS
# Average Localization Error (ALE)
Distance_List = []
for Location in C_KOO.Location:
    Mod = Location[0]
    OG = Location[1]
    
    Distance_List.append(np.sqrt((Mod[0] - OG[0])**2 + (Mod[1] - OG[1])**2))

ALE = np.sum(Distance_List)/len(Distance_List)
print('Average Localization Error (ALE) = ', ALE)

# Localization Success Ratio (LSR)
LSR = (len(Distance_List)/len(C_KOO.Undecided_Nodes_og)) * 100
print('Localization Success Ratio (LSR) = ', LSR,'%')
print('Number of Localized Nodes = ', len(Distance_List))

# No. of iterations taken
ITERS = len(C_KOO.Minimum_Fitness)
print('Number of iterations taken to Localize Nodes = ', ITERS)

# Full TIme Taken
TT = stop-start
print("Time taken to Localize = ", TT)

# Plot Test Field
C_KOO.plot_graph(False)