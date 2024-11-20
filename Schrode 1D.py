import torch
import torch.nn as nn
import numpy as np
from scipy.constants import h  # Planck's constant
from scipy import special
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 20)
        self.hidden_layer2 = nn.Linear(20, 20)
        self.hidden_layer3 = nn.Linear(20, 20)
        self.hidden_layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))  # Use `tanh`
        x = torch.tanh(self.hidden_layer2(x))
        x = torch.tanh(self.hidden_layer3(x))
        x = torch.tanh(self.hidden_layer4(x))
        x = self.output_layer(x)
        return x

# Initialize network and apply weight initialization
net = Net().to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

net.apply(init_weights)

# Define loss function and optimizer
mse_cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

# Boundary condition: v(x) = x if -a/2 <= x <= a/2, v_0 elsewhere
a = 2
v_0 = 3.41357
x_bc = np.random.uniform(low=-1.0, high=1.0, size=(500, 1)).flatten()
u_bc = np.piecewise(
    x_bc,
    [x_bc < -a / 2, (-a / 2 <= x_bc) & (x_bc <= a / 2), x_bc > a / 2],
    [v_0, lambda x: x, v_0]
)

def v(x):
    """Boundary condition function v(x)"""
    return torch.where(
        x < -a / 2, v_0, torch.where((x >= -a / 2) & (x <= a / 2), x, v_0)
    )

def f(x, net):
    """PDE loss function"""
    u = net(x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]  # First derivative
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]  # Second derivative
    pde = u_xx + (v_0 - v(x)) * u  # Adjust potential energy term here
    return pde

# Training loop
iterations = 20000
for epoch in range(iterations):
    optimizer.zero_grad()

    # Boundary condition loss
    pt_x_bc = torch.tensor(x_bc.reshape(-1, 1), dtype=torch.float32, requires_grad=False).to(device)
    pt_u_bc = torch.tensor(u_bc.reshape(-1, 1), dtype=torch.float32, requires_grad=False).to(device)
    net_bc_out = net(pt_x_bc)
    mse_bc = mse_cost_function(net_bc_out, pt_u_bc)

    # PDE loss
    x_collocation = np.random.uniform(low=-1.0, high=1.0, size=(500, 1))
    pt_x_collocation = torch.tensor(x_collocation, dtype=torch.float32, requires_grad=True).to(device)
    f_out = f(pt_x_collocation, net)
    mse_f = mse_cost_function(f_out, torch.zeros_like(f_out))

    # Total loss
    loss = mse_bc + mse_f

    # Check for NaN or Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"NaN or Inf detected at epoch {epoch}. Stopping training.")
        break

    loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True.

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    optimizer.step() #Perform a single optimization step to update parameter.

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}, BC Loss = {mse_bc.item()}, PDE Loss = {mse_f.item()}")

    if loss.item() < 0.1:  # Stop early if loss is small
        print(f"Converged at epoch {epoch} with loss {loss.item()}")
        break

# Constants
v_0 = 3.41357  # Potential energy outside the well
a = 2  # Half-width of the well
E = 3.41357  # Energy

# Define the potential function V(x)
def V(x):
    return np.where(
        x < -a / 2, v_0, np.where((x >= -a / 2) & (x <= a / 2), x, v_0)
    )

# Analytic solution using Airy functions for the time-independent Schrödinger equation
def analytic_solution(x):
    psi = np.zeros_like(x)

    # Airy functions in the inner region (-a/2 <= x <= a/2)
    inside_mask = (-a / 2 <= x) & (x <= a / 2)
    z = E - x[inside_mask]  # Argument for the Airy function (E - x)
    ai, aip, bi, bip = special.airy(z)  # Get Airy functions Ai, Ai', Bi, Bi'
    psi[inside_mask] = ai  # Use Airy Ai function

    # Exponential decay in the outer regions (x < -a/2 or x > a/2)
    outside_mask_left = x < -a / 2
    outside_mask_right = x > a / 2

    k = np.sqrt(v_0 - E)  # Decay rate (only valid for x < -a/2 or x > a/2)

    # Left side decay (exponential)
    psi[outside_mask_left] = np.exp(k * (x[outside_mask_left] + a / 2))  # Shift to match boundary

    # Right side decay (exponential)
    psi[outside_mask_right] = np.exp(-k * (x[outside_mask_right] - a / 2))  # Shift to match boundary

    return psi

# Generate x values for the analytic solution
x_values_analytic = np.linspace(-1.0, 1.0, 500)

# Calculate the analytic wavefunction
analytic_values = analytic_solution(x_values_analytic)

# Plot the solutions and probability densities
plt.figure(figsize=(12, 8))

# Plot the neural network solution
x_values_network = np.linspace(-1.0, 1.0, 500).reshape(-1, 1)
pt_x = torch.tensor(x_values_network, dtype=torch.float32, requires_grad=False).to(device)
u_values_network = net(pt_x).cpu().detach().numpy()

# Calculate probability densities
prob_density_analytic = analytic_values**2
prob_density_network = u_values_network**2

# Plot both solutions and their probability densities
plt.figure(figsize=(12, 10))

# Plot the wavefunctions
plt.subplot(2, 1, 1)
plt.plot(x_values_analytic, analytic_values, label="Analytic Solution", color="green", linestyle="--")
plt.plot(x_values_network, u_values_network, label="Network Solution", color="blue")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Wave Function Solutions")
plt.legend()
plt.grid(True)

# Plot the probability densities
plt.subplot(2, 1, 2)
plt.plot(x_values_analytic, prob_density_analytic, label="Analytic Probability Density |u(x)|^2", color="orange", linestyle="--")
plt.plot(x_values_network, prob_density_network, label="Network Probability Density |u(x)|^2", color="red")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Probability Density |u(x)|^2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the trained model
torch.save(net.state_dict(), "model_uxt.pt")
