# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:44:06 2023

@author: SHOUMYA
"""

def getMembershipRelDis(dis):
    deg = {}
    
    if dis<0 or dis>100:
        deg["l"] = 0
        deg["a"] = 0
        deg["h"] = 0
        
    elif dis>=0 and dis<30:
        deg["l"] = 1
        deg["a"] = 0
        deg["h"] = 0
        
    elif dis>=40 and dis<50:
        deg["l"] = 0
        deg["a"] = 1
        deg["h"] = 0
    
    elif dis>=60 and dis<=100:
        deg["l"] = 0
        deg["a"] = 0
        deg["h"] = 1
        
    elif dis>=30 and dis<40:
        deg["l"] = float((40-dis)/(40-30))
        deg["a"] = float((dis-30)/(40-30))
        deg["h"] = 0
    
    elif dis>=50 and dis<60:
        deg["l"] = 0
        deg["a"] = float((70-dis)/(70-50))
        deg["h"] = float((dis-50)/(70-50))
        
    return deg
        
def getMembershipSpeed(speed):
    degree = {}
    
    if speed < 0 or speed > 30:
        degree["h"] = 0
        degree["l"] = 0
    
    elif speed <10:
        degree["h"] = 0
        degree["l"] = 1
    
    elif speed>=10 and speed<20:
        degree["h"] = float((speed-10)/(20-10))
        degree["l"] = float((20-speed)/(20-10))
      
    elif speed >= 20 and speed <= 30:
        
        degree["h"] = 1
        degree["l"] = 0
        
    return degree


#input
relDis, speed = 45, 21

fuzzyRelDis=getMembershipRelDis(relDis)
fuzzySpeed=getMembershipSpeed(speed)

#Rule evaluation
val1 = max(fuzzyRelDis["l"],fuzzySpeed["h"])
val2 = fuzzyRelDis["h"]
val3 = min(fuzzyRelDis["a"], fuzzySpeed["h"])
val4 = min(fuzzyRelDis["a"], fuzzySpeed["l"])
val5 = max(val2,val4)

res, div = 0.0,0.0

for i in range(0,101,10):
    if(i<=30):
        res += (i*val5)
        div += val5
    
    elif(i<=60):
        res += (i*val3)
        div += val3
    
    elif(i<=100):
        res += (i*val1)
        div += val1

res /= div

print(res)

'''res = 0

for i in range(101):
    if i <= 30:
        if val5 > 0:
            res = max(res, i)
    elif i <= 60:
        if val3 > 0:
            res = max(res, i)
    else:
        if val1 > 0:
            res = max(res, i)

print(res)'''