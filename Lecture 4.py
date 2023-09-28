#!/usr/bin/env python
# coding: utf-8

# In[153]:


class Vehicle():
    
    def __init__(self, car_type, speed, mileage):
        self.car_type = str(car_type)
        self.speed = float(speed)
        self.mileage = int(mileage)
        
    def __str__(self):
        return f'Type:{self.car_type}, Speed:{self.speed} km/h, Mileage:{self.mileage}'
    
    #renting price based on passenger capacity
    def rent(self, passenger_capacity):
        self.passenger_capacity = passenger_capacity
        price = passenger_capacity**2
        return price
    
    #add driven mileage 
    def drive(self, added_mileage):
        self.added_mileage = int(added_mileage)
        self.mileage = self.mileage + added_mileage
        return self.mileage 
    
    #show service alert for every 10k of mileage
    def service_alert(self):
        alert = self.mileage//10000
        if self.mileage < 10000:
            print('No Service Required')
        else: 
            print('Service Required! ' * alert) 
        
        
#bus subclass with default passenger capacity 50        
class Bus(Vehicle):

    def __init__(self, car_type, speed, mileage, passenger_capacity = 50):
        super().__init__(car_type, speed, mileage)
        self.passenger_capacity = passenger_capacity
        
    def __str__(self):
        return f'Type:{self.car_type}, Speed:{self.speed} km/h, Mileage:{self.mileage}, Passenger Capacity:{self.passenger_capacity}'
    
    def rent_bus(self):
        if self.passenger_capacity > 10: 
            base_price = self.passenger_capacity**2
            price = base_price + base_price/10
            return price
    
#automobile subclass with default color 'white' 
class Auto(Vehicle):

    def __init__(self, car_type, speed, mileage, color = 'white'):
        super().__init__(car_type, speed, mileage)
        self.color = color
        
    def __str__(self):
        return f'Type:{self.car_type}, Speed:{self.speed} km/h, Mileage:{self.mileage}, Color:{self.color}'
    


# In[ ]:





# In[ ]:




