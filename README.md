# longstaff-schwartz
Implementation of the longstaff-schwartz algorithm in C++ and application to American Basket option pricing

This project is divided into three parts: 
- longstaff_schwartz.hpp contains the longstaff_schwartz function which applies the algorithm of the same name. 
- linear_regression.hpp contains the linear_regression class which allows for linear regression and calculation of statistics such as the R2 score or Cook's distance. This class is used in the Longstaff-Schwartz algorithm to approximate the continuation function.
- Finally, the last library allows for simulations on simple models such as the multidimensional Black-Scholes model.
- main.cpp contains examples of how to use each of these libraries.

To run the script you will need the standard c++11 and to include eigen3.
