#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:50:35 2023

@author: kanizfatema
"""

def getMembershipService(service):
    

def getMembershipFood(food):
    degree = {}
    
    if food < 0 or food > 1:
        degree["b"] = 0
        degree["d"] = 0
    
    elif food <=.4:
        degree["b"] = 1
        degree["d"] = 0
    
    elif food>.4 and food<.8:
        degree["b"] = float((.8-food)*(1/(.8-.4)))
        degree["d"] = float((food-.40)*1.0/(.80-.40))
      
    elif food >= .80 and food <= 1:
        
        degree["b"] = 0
        degree["d"] = 1
        
    return degree

def crispInput(val,base):
   


def ruleEvalationAssessment(service,food):
    cheap=[],average=[],generous=[]
    
    
    return cheap,average,generous


def defuzzificationAssessment(cheap,average,generous ):
    
    #cog formula  
    return cog

#input
ser, foodd = 6, 3

fuzzyservice=getMembershipService(s)
fuzzyfood=getMembershipFood(f)


cheap,average,generous = ruleEvalationAssessment(s,f)



conAssessment = defuzzificationAssessment(cheap,average,generous )
print("Fuzzified Continuous Assessment: ",conAssessment)