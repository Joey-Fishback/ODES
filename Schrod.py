import torch
import torch.nn as nn
import scipy.constants as const

from torch.autograd import Variable  # Provides support for automatic differentiation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Checks if a GPU is available; if not, it defaults to CPU

import numpy as np  # Import NumPy for numerical operations

# We consider Net as our solution u_theta(x,t)

"""
The neural network, Net, is defined to approximate the solution u_theta(x, t).
Inputs: x (space), t (time)
Output: u (dependent variable)
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # Initializes the parent `nn.Module`
        # Define layers: input layer (2 -> 5), 5 hidden layers (5 -> 5), and an output layer (5 -> 1)
        self.hidden_layer1 = nn.Linear(2, 5)  # Fully connected layer: 2 inputs, 5 outputs
        self.hidden_layer2 = nn.Linear(5, 5)
        self.hidden_layer3 = nn.Linear(5, 5)
        self.hidden_layer4 = nn.Linear(5, 5)
        self.hidden_layer5 = nn.Linear(5, 5)
        self.output_layer = nn.Linear(5, 1)  # Final output: 5 inputs, 1 output

    def forward(self, x, t):
        # Reshape x and t to ensure 2D inputs
        x = x.view(-1, 1)  # Ensures [batch_size, 1]
        t = t.view(-1, 1)
        # Combine x and t into a single tensor of shape [batch_size, 2]
        inputs = torch.cat([x, t], axis=1)

        # Pass data through hidden layers with sigmoid activations
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))

        # Output layer (no activation for regression tasks)
        output = self.output_layer(layer5_out)
        return output


### (2) Model
net = Net()  # Instantiate the neural network
net = net.to(device)  # Move the model to GPU (if available)

# Loss function: Mean Squared Error (MSE) for regression tasks
mse_cost_function = torch.nn.MSELoss()

# Optimizer: Adam optimization algorithm
optimizer = torch.optim.Adam(net.parameters())


# Constant initial condition: u(x, 0) = 1/sqrt(a)
a = 2
C = pow(a,1/2)  # Define the constant value
x_ic = np.linspace(-1.0, 1.0, 500).reshape(-1, 1)  # Spatial points for IC
t_ic = np.zeros_like(x_ic)  # Time is fixed at t = 0
u_ic = np.full_like(x_ic, C)  # Constant function u(x, 0) = C



## Data from Boundary Conditions
# Boundary condition: v(x) = x if  -a/2 <= x <= a/2, v_0 elsewhere
x_bc = np.random.uniform(low=-1.0, high=1.0, size=(500, 1))  # Generate 500 random x values
t_bc = np.zeros((500, 1))  # Time is 0 for the boundary condition
x_bc = x_bc.flatten()  # Flatten x_bc for piecewise operation
v_0 = 3.41357
a = 2
# Compute boundary condition values using `np.piecewise`
u_bc = np.piecewise(x_bc,
                    [x_bc < -a/2, (-a/2 <= x_bc) & (x_bc <= a/2), x_bc > a/2],
                    [v_0, lambda x: x, v_0]

                    )
def v(x):
    """Boundary condition function v(x)"""
    # Using torch.where to handle piecewise conditions for tensor inputs
    return torch.where(x < -a/2, v_0,
                       torch.where((x >= -a/2) & (x <= a/2), x, v_0))

def f(x, t, net):
    # Compute u from the neural network
    u = net(x, t)
    # Calculate derivatives using `torch.autograd.grad`
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]  # First-order spatial derivative
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0] # Second-order spatial derivative
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]  # First-order spatial derivative

    h = const.h
    m = 4/pow(h,2)
    # Define the DE: f = u_xx + h^2/2m(3.41357 - u) * u

    pde =  u_xx + (2*m)/pow(h,2) * (3.41357 - v(x)) * u
    return pde



### (3) Training / Fitting
iterations = 20000
previous_validation_loss = 99999999.0  # Initialize large loss value
for epoch in range(iterations):
    optimizer.zero_grad()  # Reset gradients

    # Loss for initial condition

    # Convert IC data to PyTorch tensors
    pt_x_ic = Variable(torch.from_numpy(x_ic).float(), requires_grad=False).to(device)
    pt_t_ic = Variable(torch.from_numpy(t_ic).float(), requires_grad=False).to(device)
    pt_u_ic = Variable(torch.from_numpy(u_ic).float().view(-1, 1), requires_grad=False).to(device)


    # Compute network output for initial condition points
    net_ic_out = net(pt_x_ic, pt_t_ic)

    # Compute MSE loss for initial condition
    mse_ic = mse_cost_function(net_ic_out, pt_u_ic)

    # Boundary condition loss

    # Convert IC data to PyTorch tensors
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float().view(-1, 1), requires_grad=False).to(
        device)  # Reshape to [500, 1]

    # Compute network output for initial condition points

    net_bc_out = net(pt_x_bc, pt_t_bc)

    # Compute network output for initial condition points
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)

    # PDE loss

    x_collocation = np.random.uniform(low=0.0, high=1.0, size=(500, 1))  # Random x values for collocation points
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500, 1))  # Random t values for collocation points
    all_zeros = np.zeros((500, 1))  # Zero target values for f


    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = f(pt_x_collocation, pt_t_collocation, net)  # PDE output
    mse_f = mse_cost_function(f_out, pt_all_zeros)  # Compute MSE loss for PDE

    # Total loss: BC loss + PDE loss
    loss = mse_ic + mse_u + mse_f
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters

    with torch.autograd.no_grad():
        print(epoch, "Training Loss:", loss.data)
    if loss.detach().numpy() < 0.07:  # Stop early if loss is sufficiently small
        break

# Visualization of the solution
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from torch.autograd import Variable

# Generate data for visualizations
x = np.linspace(-2, 2, 500).reshape(-1, 1)  # Spatial points
time_slices = [-0.5, 0, 0.5]  # Time slices for the first plot
x_grid = np.linspace(-2, 2, 200)  # Grid for heatmap and 3D
t_grid = np.linspace(-1, 1, 200)

# Flatten and convert to tensors
pt_x_grid = Variable(torch.from_numpy(x_grid).float(), requires_grad=False).to(device)
pt_t_grid = Variable(torch.from_numpy(t_grid).float(), requires_grad=False).to(device)

# Prepare the grid for heatmap and 3D plots
ms_x, ms_t = np.meshgrid(x_grid, t_grid)
x_flat = ms_x.ravel().reshape(-1, 1)
t_flat = ms_t.ravel().reshape(-1, 1)

pt_x_flat = Variable(torch.from_numpy(x_flat).float(), requires_grad=False).to(device)
pt_t_flat = Variable(torch.from_numpy(t_flat).float(), requires_grad=False).to(device)

# Compute values from the neural network
pt_u_flat = net(pt_x_flat, pt_t_flat)
psi_flat = pt_u_flat.data.cpu().numpy().reshape(ms_x.shape)
prob_density_flat = np.abs(psi_flat) ** 2

# Initialize a figure with 3 subplots
fig = plt.figure(figsize=(18, 10))

# Plot 1: Probability density for specific time slices
ax1 = fig.add_subplot(2, 2, 1)
for t_value in time_slices:
    t = np.full_like(x, t_value)  # Time array for the slice
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)
    pt_u = net(pt_x, pt_t)
    psi = pt_u.data.cpu().numpy()
    prob_density = np.abs(psi) ** 2
    ax1.plot(x, prob_density, label=f"$t={t_value}$")
ax1.set_title(r"Probability Density $|\psi(x,t)|^2$")
ax1.set_xlabel(r"Position $x$")
ax1.set_ylabel(r"Probability Density $|\psi|^2$")
ax1.legend()
ax1.grid(True)

# Plot 2: Heatmap of probability density
ax2 = fig.add_subplot(2, 2, 2)
heatmap = ax2.contourf(ms_x, ms_t, prob_density_flat, levels=50, cmap="inferno")
fig.colorbar(heatmap, ax=ax2, label=r"Probability Density $|\psi|^2$")
ax2.set_title(r"Heatmap of $|\psi(x, t)|^2$")
ax2.set_xlabel(r"Position $x$")
ax2.set_ylabel(r"Time $t$")

# Plot 3: 3D surface plot of wavefunction and probability density
ax3 = fig.add_subplot(2, 1, 2, projection='3d')
# Plot wavefunction
ax3.plot_surface(ms_x, ms_t, psi_flat, cmap='coolwarm', alpha=0.6)
# Plot probability density
ax3.plot_surface(ms_x, ms_t, prob_density_flat, cmap='viridis', alpha=0.8)
ax3.set_title("Wavefunction and Probability Density")
ax3.set_xlabel("Position $x$")
ax3.set_ylabel("Time $t$")
ax3.set_zlabel(r"$\psi$ and $|\psi|^2$")

# Show the combined plot
plt.tight_layout()
plt.show()

# Save the trained model
torch.save(net.state_dict(), "model_uxt.pt")
