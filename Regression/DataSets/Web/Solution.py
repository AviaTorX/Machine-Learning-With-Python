import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

print(data.shape)
