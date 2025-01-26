import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define parameters

A = 0.50
B = 0.0

# Define initial conditions
U0 = 0.1
y0 = 0

# Define the system of ODEs
def system(t, state):
    U, y = state
    dU_dt = y
    dy_dt = -f1 * U**3 + f2 * U + A * np.cos(B * t)
    return [dU_dt, dy_dt]

# Time span for the simulation
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Initial conditions
initial_conditions = [U0, y0]

# Solve the system
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extract solutions
t = solution.t
U = solution.y[0]
y = solution.y[1]

# Plot the results
fig=plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(U, y, t, label=f'IC: U0={U0}, y0={y0}')
# Labels and title
ax.set_xlabel('y')
ax.set_ylabel('U')
ax.set_zlabel('time')
ax.set_title('Chaotic Behaviour')
ax.legend()
plt.show()
plt.plot(U,y)
plt.xlabel('U')
plt.ylabel('y')
plt.grid(True)
plt.show()
