# Black-Litterman-Model

#### Brief Introduction
1. Implement Black-Litterman Model using **Python**.  
2. Use **4 different** kinds of view type to evaluate Black-Litterman Model.
3. Implement **back-test** by stock market.
4  Plot line charts which display accumulated return using BL model vs that using eqaul weight (comparative approach) for these 4 types. 
5. Data: price of 10 stocks in the US stock market during the past ten years.

*Data Source: Wind*

#### Details
1. 4 different kinds of view type:  
  * **Market value as view**:    
  It uses weights of 10 stocks' market value as weights of assets allocation.   
  * **Arbitrary views**:    
  It measures the result when views are given arbitrarily and inaccurately.   
  * **Reasonable views**:  
  It measures the result when views are given reasonably and accurately.  
  * **Near period return as view**:  
  It measures the result when stock price and return of nearest T periods are used as views.  

#### Results
1. These 4 kinds of view type show results as follows:  
  * **Market value as views**:    
    * Nearly equal performance as Equal Weight method (comparative approach).
    * Market value weight can not predict future return of stock accurately.
  * **Arbitrary views**:    
    * Nearly equal performance as Equal Weight method (comparative approach).
    * It can not make money if no strong economic knowledge and efficient views even if using a complex model like BL.
  * **Reasonable views**:  
    * BL Model is really **strong** when the views are **efficient and accurate**.
    * It performs much better when whole market goes up largely. (e.g. 4 huge growth in year 2015)
    * But, it can not resist the large drop (e.g. 3 large drop in year 2015) within a short time. 
  * **Near period return as views**:  
    * Views which generated from nearest data can response efficiently and quickly to huge change within a short time. 
    * It performs well when the whole market goes down (e.g. Two large drops in year 2015).

