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
from scipy.optimize import brentq
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Implied Volatility - Interactive Heatmap")
st.text("Demonstrates how Implied Volatility of a stock varies with Time to Maturity and the Strike Price of the option")

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
            return v0
        v = v0
    return v

def implied_vol_put(S, K, v, t, r, mp):
    count = 0
    for count in range (1, 51): #max iteration set to 50
        P = bs_put(S, K, v, t, r)
        veg = vega(S, K, v, t, r)
        v0 = v - (P - mp)/(veg)
        
        if abs(v0 - v) < 1e-4:
            return v0
        v = v0
    return v

def option_button():
    types = ["Call", "Put"]
    choice = st.segmented_control(
        "Option Type", types,
        selection_mode = "single"
    )
    return choice

def data_frame_call(min_K, max_K, S, v, r, mp):
    n = 8
    strike_values = np.linspace(min_K, max_K, n)
    time_values = np.linspace(0.5, 4.0, n)

    data = np.empty((n, n), dtype=float)

    for i in range(len(time_values)):
        t = time_values[i]
        for j in range(len(strike_values)):
            K = strike_values[j]
            iv = implied_vol_call(S, K, v, t, r, mp)
            data[i,j] = iv
    strike_values.round(2, out=strike_values)
    time_values.round(2, out=time_values)
    
    df_heatmap = pd.DataFrame(data, index = time_values, columns = strike_values)
    return df_heatmap
    
def data_frame_put(min_K, max_K, S, v, r, mp):
    n = 8
    strike_values = np.linspace(min_K, max_K, n)
    time_values = np.linspace(0.5, 4.0, n)

    data = np.empty((n, n), dtype=float)

    for i in range(len(time_values)):
        t = time_values[i]
        for j in range(len(strike_values)):
            K = strike_values[j]
            iv = implied_vol_put(S, K, v, t, r, mp)
            data[i,j] = iv
    strike_values.round(2, out=strike_values)
    time_values.round(2, out=time_values)

    df_heatmap = pd.DataFrame(data, index = time_values, columns = strike_values)
    return df_heatmap

def heatmap_plot(df_heatmap, choice):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, cbar_kws={'label': 'Implied Volatility'})
    ax.set_title(choice)
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Time to Maturity (years)")
    st.pyplot(fig)                           
    plt.close(fig)  

def main():
    choice = option_button()

    with st.sidebar:
        st.markdown("# 📊 Implied Volatility Heatmap")

        S = st.number_input("Spot Price (S)", min_value = 0.0,
                            max_value = None, value = 100.0, 
                            step = 1.)

        k_lo_allowed = 0.4 * S
        k_hi_allowed = 1.6 * S

        # default range centered around spot
        default_lo = max(k_lo_allowed, 0.8 * S)
        default_hi = min(k_hi_allowed, 1.2 * S)

        strike_range = st.slider(
            "Strike range",
            min_value=float(k_lo_allowed),
            max_value=float(k_hi_allowed),
            value=(float(default_lo), float(default_hi)),
            step=1.0,
            help="Pick the min/max strike for the heatmap."
        )
        min_K, max_K = strike_range
    
        r = st.number_input("Risk-free interest rate (r)", 
                            min_value = 0.0,
                            max_value = 0.2, 
                            value = 0.01, 
                            step = 0.01)
        
        mp = st.number_input("Option Market Price", 
                            value = 25.0, 
                            step = .1)
        
    if choice == "Call":
        df_heatmap = data_frame_call(min_K, max_K, S, v, r, mp)
        heatmap_plot(df_heatmap, choice)
    elif choice == "Put":
        df_heatmap = data_frame_put(min_K, max_K, S, v, r, mp)     
        heatmap_plot(df_heatmap, choice)
                
    st.markdown(
                """
                <div style="color: lightblue; background-color: Steelblue; padding: 10px; border-radius: 5px;">
                    This version uses the Newton-Raphson Iterative \
                    method to calculate Implied Volatility. The method \
                    has issues with convergence, especially if the initial guess is far.  \
                    The reason I've used this method is because Newton-Raphson often appears  \
                    in undergraduate study and I thought it was worth adding to demonstrate its limitations.  \
                    Here is the link to my Implied Volatility Heatmap which uses Brent's method.
                </div>
                """,
                unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
# %%
