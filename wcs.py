#!/usr/bin/env python

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict

words = "apple banana apple strawberry banana lemon"

d = defaultdict(int)
for word in words.split():
	d[word] += 1
	print d
