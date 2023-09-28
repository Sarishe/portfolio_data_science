#!/usr/bin/env python
# coding: utf-8

# In[16]:


import math
import random

def random_generator():
    #generate a list of 100 random float numbers
    list_random = []
    for i in range(1,100):
        list_random.append(random.uniform(1,100))
    
    #calculate sum and average of those random numbers
    summed = math.fsum(list_random)
    average = summed/100
    listed = [summed, average]
    return listed

#call 'random' function 100 times and slice the lists into 
slicy = []
for i in range(1,100):
    slicy.append(random_generator())

slicy_united = sum(slicy, [])
list_of_sums = slicy_united[::2]
list_of_averages = slicy_united[1::2]
print('List of sums: \n', list_of_sums, '\n \n List of averages: \n', list_of_averages)

#numbers do not deviate too much from each other, approaching normal distribution

