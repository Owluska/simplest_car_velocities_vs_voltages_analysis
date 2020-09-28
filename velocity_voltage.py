# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:32:25 2020

@author: User
"""

import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np



forward = {'5.0': [56, 56.5, 57.5, 57.1],
           '5.5': [66.8, 68.2, 67.3, 67.8, 67.8, 68.5],
           '6.0': [81.1, 82.0, 80.9, 72.8, 74.5, 76.2, 75.1, 76.7, 73.5, 74.3, 72.1, 72.0, 72.1, 73.0],
           '6.5': [86.7, 85.0, 82.5, 81.3, 81.3, 83.5, 86.5, 81.3, 80.0, 79.5, 80.5, 79.5, 79.5],
           '7.0': [88.0, 87.5, 98.5, 100.0, 91.0, 100.0, 100.5, 100.5],
           '7.5': [107.0, 110.0, 111.0, 108.0, 110.0, 110.0],
           '7.8': [107.0, 110.0, 110.0]}

backward = {'5.0': [52.5, 103-52.5, 154-103, 199-154, 245-199, 288-245],
            '5.5': [57.0, 113-57.0, 170-113, 228-170, 285-228, 346.5-285],
            '6.0': [64.5, 129.5-64.5, 195-129.5, 261-195, 334-261, 404-334],
            '6.5': [79.0, 155.8-79.0, 230.0-155.8, 308.5-230.0, 389.5-308.5, 468.0-389.5],
            '7.0': [82.2, 166-82.2, 250-166, 335-250, 421-335, 81],
            '7.5': [93.5, 191.2-93.5, 286-191.2, 382-286, 91.2, 183-91.2, 277-183],
            '8.0': [97.0, 198-97, 299-198, 403-299, 100, 200-100, 301-100]}

right = {'5.0': [25, 20, 15, 18, 22, 16, 14],
         '5.5': [30, 22, 26, 30, 25],
         '6.0': [114, 113, 114, 98, 89, 90, 89, 89],
         '6.5': [79, 75, 85, 79, 86, 90, 86],
         '7.0': [84, 74, 72, 80, 87],
         '7.5': [75, 71, 64, 60, 58, 59, 62],
         '7.8': [54, 64, 43, 49, 44, 50]}


forward_voltages = [float(k) for k in forward.keys()]
forward_mean, forward_median =[], []
for f in forward:
    forward_mean.append(np.mean(forward[f]))
    forward_median.append(np.median(forward[f]))
    
backward_voltages = [float(k) for k in backward.keys()]

backward_mean, backward_median =[], []
for f in backward:
    backward_mean.append(np.mean(backward[f]))
    backward_median.append(np.median(backward[f]))
    
right_voltages = [float(k) for k in right.keys()]
right_std = [np.std(right[r]) for r in right]
right_mean = [np.mean(right[r]) for r in right]

# plt.errorbar(right_voltages, right_mean, right_std, linestyle='None', marker='^')
# plt.grid(True)
# plt.title('Angular velocity vs supply voltage')
# plt.xlabel('Voltage, V') 
# plt.ylabel('Mean angle, \N{DEGREE SIGN}')

Xf = np.matrix(forward_voltages).T
yf = np.array(forward_mean)
reg_f = linear_model.LinearRegression().fit(Xf, yf)
# print('Regression coefficients for forward velocitiy: {}'.format(reg_f.coef_))
_yf = reg_f.predict(Xf)


Xb = np.matrix(backward_voltages).T
yb = np.array(backward_mean)
reg_f = linear_model.LinearRegression().fit(Xb, yb)
# print('Regression coefficients for forward velocitiy: {}'.format(reg_f.coef_))
_yb = reg_f.predict(Xb)  

plt.plot(forward_voltages, forward_mean) 
plt.plot(forward_voltages, forward_median)
plt.plot(forward_voltages, _yf)

plt.plot(backward_voltages, backward_mean) 
plt.plot(backward_voltages, backward_median)
plt.plot(backward_voltages, _yb)

plt.title('Velocity vs supply voltage')
plt.xlabel('Voltage, V') 
plt.ylabel('Distance, cm')
plt.legend(['Forward mean values', 'Forward median values', 'Forward regression',
            'Backward mean values', 'Backward median values', 'Backward regression'])

