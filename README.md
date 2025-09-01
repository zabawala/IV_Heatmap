# Implied Volatility Heatmap 
This repository contans code for an implied volatility heatmap with 2 different solvers. The project visualizes implied volatility (IV) as a function of strike price and time to maturity for European options under the Black–Scholes model. The app is built with Python and Streamlit, and demonstrates the performance of two different numerical solvers for computing IV:

- **Newton–Raphson method** (fast, but sensitive to initial guess and Vega).  
- **Brent’s method** (robust root-finder, guarantees convergence if a solution exists). 

---

## 📈 Features  

- Interactive **heatmap** of implied volatilities across a user-defined range of strikes and maturities.  
- User inputs:  
  - Spot price  
  - Strike price range  
  - Time to maturity (0.5–4 years)  
  - Market option price  
  - Risk-free rate  

  ## 🌐 Live Apps  

- **Newton–Raphson IV Heatmap**  
  👉 [Link to app](https://ivheatmap-newton.streamlit.app)  

- **Brent’s Method IV Heatmap**  
  👉 [Link to app](https://ivheatmap-brents.streamlit.app)  
