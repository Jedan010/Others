# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:20:28 2017

@author: J
"""


from sympy import *

x ,y, k ,b = symbols('x y k b')
f = Function('f')
g = Function('g')
h = Function('h')


f = x**2/16 + y**2/9 -1 
g = k*x + b -y
y0 = solve(f, y)
print('y:' , y0)

h = g.subs(y, y0[0])
print('h:' , h)

k0 = solve(h, k)
b0 = solve(h, b)
print('k:',k0)
print('b:',b0)


from sympy.parsing.sympy_parser import parse_expr
from sympy import plot_implicit
ezplot = lambda expr: plot_implicit(parse_expr(expr))
ezplot('3*sin(x) - y')
