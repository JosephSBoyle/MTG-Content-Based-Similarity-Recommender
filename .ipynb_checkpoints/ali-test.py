import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
import random
import numpy as np

X = [6278, 3113, 5236, 11584, 12628, 7725, 8604, 14266, 6125, 9350, 3212, 9003, 3523, 12888, 9460, 13431, 17809, 2812, 11825, 2398]

#
# def lifetime(x, delta=2398, beta=6165.5):
#     if x <= delta:
#         return 1
#     else:
#         return (1/beta) * np.exp(-x/beta)
#
# n = 100_000
# x = [i for i in range(n)]
# y = [lifetime(i) for i in range(n)]
#
# integral = scipy.integrate.trapezoid(y)
# print(integral * n / (n + 1))
# p_lifetime = {t <= delta: 1, x > delta: (1/Beta) * e^-x/Beta}



hours = range(1, 20_000)
remaining_li = []
for hour in hours:
    remaining = len(X)
    for i in X:
        if i > hour:
            pass
        else:
            remaining -= 1
    remaining_li.append(remaining)

print(len(hours), len(remaining_li))
plt.scatter(hours, remaining_li)
plt.ylabel('remaining ball-bearings')
plt.xlabel('hours surpassed')
plt.show()

loc, scale = scipy.stats..fit(X)
print(loc, scale)