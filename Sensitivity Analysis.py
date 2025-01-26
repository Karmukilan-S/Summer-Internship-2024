import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha=1
omega1=1
omega2=1
kappa1=1
kappa2=0.5
lambda1=1
lambda2=1
mew2=2
f1=2*alpha/((kappa1)**2+(lambda1)**2-(omega1)**2)
f2=((kappa2)**2+(lambda2)**2-(omega2)**2+2*mew2)/((kappa1)**2+(lambda1)**2-(omega1)**2)

A=0.5
B=0.175
#derivatives for sensitivity analysis
def derivatives(t, U, y):
    dU_dt = y
    dy_dt = -f1 * U**3 + f2 * U
    return dU_dt, dy_dt

#derivatives for analysing chaos in system
def derivatives_chaos(t, U, y):
    dU_dt = y
    dy_dt = -f1 * U**3 + f2*U +  A*np.cos(B*t)
    return dU_dt, dy_dt
print(derivatives_chaos(0, 0, 0))
def runge_kutta(t0, U0, y0, t_end, dt):
    t_values = np.arange(t0, t_end, dt)
    U_values = []
    y_values = []

    U = U0
    y = y0

    for t in t_values:
        U_values.append(U)
        y_values.append(y)

        k1_U, k1_y = derivatives(t, U, y)
        k2_U, k2_y = derivatives(t + 0.5*dt, U + 0.5*dt*k1_U, y + 0.5*dt*k1_y)
        k3_U, k3_y = derivatives(t + 0.5*dt, U + 0.5*dt*k2_U, y + 0.5*dt*k2_y)
        k4_U, k4_y = derivatives(t + dt, U + dt*k3_U, y + dt*k3_y)

        U += (dt/6) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
        y += (dt/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

    return t_values, U_values, y_values


U0 =0.1
y0 = 0.0
t0 = 0.0
t_end = 100.0
dt = 0.01


t_values, U_values, y_values = runge_kutta(t0, U0, y0, t_end, dt)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t_values, U_values, label='U(t)', color='blue')
plt.plot(t_values, y_values, label='y(t)', color='orange')
plt.xlabel('Time (t)')
plt.ylabel('U(t) and y(t)')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
