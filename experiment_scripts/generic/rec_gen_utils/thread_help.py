import sys
import os
from multiprocessing import Pool

def mute():
    sys.stdout = open(os.devnull, 'w')    
    sys.stderr = open(os.devnull, 'w')    