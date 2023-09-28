#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. За допомогою def() та return написати чисту функцію, яка створить список з N послідовних натуральних чисел, і поверне список з квадратів цих натуральних чисел. 
def squared(N):
    list1 = []
    for i in range(1,N+1):
        list1.append(i)
    return list([i**2 for i in list1])

N = int(input('By which number to square: '))
print(squared(N))


# In[2]:


#2. Зробити попередній пункт за допомогою оператора map. 
def squared(N):
    return list(map(lambda i: i**2, range(1, N+1)))

N = int(input('By which number to square: '))
print(squared(N))


# In[2]:


#2. Зробити попередній пункт за допомогою оператора map. варіант 2
def squared(N):
    return N**2

N = int(input('By which number to square: '))
print(list(map(squared, range(1, N+1))))


# In[23]:


#3. Зробити попередній пункт за допомогою розуміння списків
def squared(N):
    return list(i**2 for i in range(1, N+1))

N = int(input('By which number to square: '))
print(squared(N))


# In[1]:


#4. За допомогою def() та return написати чисту функцію, яка створить список з N послідовних натуральних чисел і поверне список чисел, які є квадратами якогось натурального числа.
import math
def squared(N):
    x = math.sqrt(N)
    if x.is_integer():
        return N
    else:
        pass

N = int(input('By which number to check squares: '))
print(list(filter(squared, range(1, N+1))))


# In[ ]:


#5. Написати лямбда функцію, яка перевірятиме належність числа N до відрізку [a, b]
N = int(input('Which number would you like to check if it belongs to a range? '))
a = int(input('What is the starting point of the range? '))
b = int(input('What is the last point of the range? '))
check = lambda i: True if (i in range (a,b)) else False
print(check(N))


# In[ ]:




