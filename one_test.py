# -*- coding: utf-8 -*-

import random
import thinkstats2
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#from pandas import DataFrame, Series

## There is only one test
# http://allendowney.blogspot.co.uk/2011/05/there-is-only-one-test.html

## Example casino problem

# Suppose you run a casino and you suspect that a customer has replaced a die 
# provided by the casino with a ``crooked die''; that is, one that has been 
# tampered with to make one of the faces more likely to come up than the others.
# You apprehend the alleged cheater and confiscate the die, but now you have to
# prove that it is crooked.  You roll the die 60 times and get the following results:
    
# Value       1    2    3    4    5    6 
# Frequency   8    9   19    6    8   10 

# What is the probability of seeing results like this by chance?

# H_0 = die is fair
# H_1 = die is not fair

# To compute a p-value use chi-squared statistic: 
# for each value we compare the expected frequency, 'exp', and the observed 
# frequency, 'obs', and compute the sum of the squared relative differences:
    
sides = 6
num_rolls = 60
odds = num_rolls/sides                       
exp = [odds]*60                # even odds for each side of die
obs = [8, 9, 19, 6, 8, 10]
        
def ChiSquared(expected, observed):
    total = 0.0
    for x, exp in expected.Items():
        obs = observed.Freq(x)
        total += (obs - exp)**2 / exp
    return total
    
## Simulation code
    
def SimulateRolls(sides, num_rolls):
    """Generates a Hist of simulated die rolls.
    
    Args:
      sides: number of sides on the die
      num_rolls: number of times to rolls


    Returns:
      Hist object
    """
    hist = Pmf.Hist()
    for i in range(num_rolls):
        roll = random.randint(1, sides)
        hist.Incr(roll)
    return hist

# Runs 1000 simulations

count = 0.
num_trials = 1000
num_rolls = 60
threshold = ChiSquared(exp, obs)


for _ in range(num_trials):
    simulated = SimulateRolls(sides, num_rolls)
    chi2 = ChiSquared(exp, simulated)
    if chi2 >= threshold:
        count += 1

        pvalue = count / num_trials
        print('p-value', pvalue)