# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:iv_heatmap] *
#     language: python
#     name: conda-env-iv_heatmap-py
# ---

# %%
import math
from scipy.stats import norm
import streamlit as st

st.write("Hello World")

v = 0.2 # Standard guess for volatility

# ---------------------------------
# Math functions used in solvers
# ---------------------------------

def d1_d2(S, K, v, t, r):
    d1 = (math.log(S/K) + t*(r + (0.5 * v ** 2)))/(v * math.sqrt(t))
    d2 = d1 - (v * math.sqrt(t))
    return d1, d2
    
def bs_call(S, K, v, t, r):
    d1, d2 = d1_d2(S, K, v, t, r)
    C = (S * norm.cdf(d1)) - (K * math.exp(-r * t) * norm.cdf(d2))
    return C

def bs_put(S, K, v, t, r):
    d1, d2 = d1_d2(S, K, v, t, r)
    P = (K * math.exp(-r * t) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return P

def vega(S, K, v, t, r):
    d1, d2 = d1_d2(S, K, v, t, r)
    return (S * norm.pdf(d1) * math.sqrt(t))

# ---------------------------------
# Implied volatility solvers
# ---------------------------------

def implied_vol_call(S, K, v, t, r, mp):
    count = 0
    for count in range (1, 51): #max iteration set to 50
        C = bs_call(S, K, v, t, r)
        veg = vega(S, K, v, t, r)
        v0 = v - (C - mp)/(veg)
        
        if abs(v0 - v) < 1e-4:
            return v0, count
        v = v0
    return v, count

def implied_vol_put(S, K, v, t, r, mp):
    count = 0
    for count in range (1, 51): #max iteration set to 50
        P = bs_put(S, K, v, t, r)
        veg = vega(S, K, v, t, r)
        v0 = v - (P - mp)/(veg)
        
        if abs(v0 - v) < 1e-4:
            return v0, count
        v = v0
    return v, count

def option_type():
    while True:
        choice = input("Option type: (Call/Put)")
        
        if choice == "Call":
            break
            
        elif choice == "Put":
            break
            
        else:
            print("Invalid input. Please enter 'Call' or 'Put'. (case sensitive)")
            
def main():
    choice = option_type()
    S = float(input('Spot Price:'))
    K = float(input('Strike Price:'))
    mp = float(input('Option market price: '))
    t = float(input('Time to expiry:'))
    r = float(input('Risk-free interest rate:'))
    if choice == "Call":
        iv, count = implied_vol_call(S, K, v, t, r, mp)
        print("Implied volatility: ", round(iv, 5))
        print("Number of iterations: ", count)
    else:
        iv, count = implied_vol_put(S, K, v, t, r, mp)
        print("Implied Volatility: ", round(iv, 5))
        print("Number of iterations: ", count)

main()

# %%
