"""

"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import corner

from scipy.stats import gaussian_kde
from scipy.special import sph_harm

from os.path import dirname
FILE_DIR = dirname(__file__)

import torch
from torch.utils.data import DataLoader, TensorDataset

from CNF import *

def visualize_samples(x_0:np.ndarray, z:np.ndarray, title):
    plt.figure(figsize=(16, 6))
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2

    plt.subplot(1, 3, 1)
    plt.scatter(x_0[:, 0], x_0[:, 1], alpha=0.5, label='Uniform Samples')
    plt.title('Base Distribution (Uniform) $p_U(\mathbf{u})$')
    plt.xlabel('$u_1$')
    plt.ylabel('$u_2$')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.subplot(1, 3, 2)
    plt.quiver(x_0[:, 0], x_0[:, 1], 
               z[:, 0]-x_0[:, 0], z[:, 1]-x_0[:, 1], 
               headwidth=3, 
               headaxislength=3,
               headlength=3,
               scale=2)
    plt.scatter(x_0[:, 0], x_0[:, 1], alpha=0.5, color='blue', label='Uniform Samples')
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5, color='red', label='Transformed Samples')
    plt.title(title + " $p_X(\mathbf{x})=T(\mathbf{u})$")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.subplot(1, 3, 3)
    kde = gaussian_kde(z.T, bw_method='scott')
    # xmin, xmax = z[:, 0].min(), z[:, 0].max()
    # ymin, ymax = z[:, 1].min(), z[:, 1].max()
    xgrid = np.linspace(xmin, xmax, 100)
    ygrid = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.title(r"Smoothed Plot of $T: p_U(\mathbf{u}) \rightarrow p_X(\mathbf{x})$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.show()

def plot_trajectories(traj:np.ndarray, n = 2000):
    """Plot trajectories of some selected samples."""

    traj = traj[:, :n, :]
    n_times, n_points, _ = traj.shape
    color1 = (163/255,96/255,130/255)
    color2 = (76/255,114/255,176/255)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [color1, color2])
    clrs = np.linspace(0, 1, n_points)
    clrs_matrix = np.tile(clrs, (n_times, 1))
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :, 0], traj[0, :, 1], s=5, alpha=0.8, c=[color1])
    plt.scatter(traj[:, :, 0], traj[:, :, 1], s=0.2, alpha=0.2, c=clrs_matrix.T, cmap=cmap)
    # plt.scatter(x_vals, y_vals, s=0.2, alpha=0.2, c=clrs, cmap=cmap)
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, alpha=0.9, c=[color2])
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_smooth_dist(traj, nsamples):
    z = traj[-1, :nsamples]
    kde = gaussian_kde(z[:,:2].T, bw_method='scott')
    xmin, xmax = z[:, 0].min(), z[:, 0].max()
    ymin, ymax = z[:, 1].min(), z[:, 1].max()
    xgrid = np.linspace(xmin, xmax, 100)
    ygrid = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    plt.contourf(X, Y, Z, levels=100, cmap="viridis")
    plt.title(r"Smoothed Plot of $T: p_U(\mathbf{u}) \rightarrow p_X(\mathbf{x})$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.show()



def harm_labels(ndim):
    """
    Generates harmonic labels for plotting
    """
    nharm = int(ndim**0.5)
    labels = []
    for i in range(nharm):
        labels.append(f'$k_{{{i}0}}$')
        for j in range(1, i + 1):
            labels.append(f'$Re(k_{{{i}{j}}})$')
            labels.append(f'$Im(k_{{{i}{j}}})$')
    return labels    
    

def _plot(samples, ndim, **kwargs):
    """
    """

    quantiles = kwargs.get("quantiles", [0.05, 0.5, 0.95])
    k_true = kwargs.get("k_true", None)
    if k_true is None: 
        k_true = np.zeros(ndim)

    means = np.mean(samples, axis=0)
    medians = np.median(samples, axis=0)

    corner_plot_kwargs = {
        "smooth": 1.0, 
        "range": kwargs.get("range", None), 
        "quantiles": quantiles, 
        "labels": harm_labels(ndim),
        "show_titles": True, 
        "title_fmt": kwargs.get("title_fmt", '.2E'),
        "title_kwargs": kwargs.get("title_kwargs", {"fontsize": 12, "color": "red"}),
        "verbose": kwargs.get("verbose", True)
    }

    
    legend_elements = [Line2D([0], [0], color='g', marker='o', linestyle='None', label='Mean'),
                   Line2D([0], [0], color='b', marker='o', linestyle='None', label='Median')]
    
    fig_corner:figure.Figure = corner.corner(samples, **corner_plot_kwargs)
    fig_axes = np.array(fig_corner.axes).reshape((ndim, ndim))
    for yi in range(ndim):
        for xi in range(yi):
            ax:plt.Axes = fig_axes[yi, xi]
            ax.axvline(k_true[xi], color='r')
            ax.axhline(k_true[yi], color='r')
            ax.plot(k_true[xi], k_true[yi], "sr")
            ax.plot(means[xi], means[yi], "sg")
            ax.plot(medians[xi], medians[yi], "sb")
    fig_corner.legend(handles=legend_elements, loc='upper right')

    savefile = kwargs.get("savefile", None)
    if savefile is not None:
        fig_corner.savefig(savefile)
    plt.show()

def plot_data(ndim, samples, **kwargs):
    """
    
    """
    res = None
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().numpy()

    plot_kwargs = {**kwargs, "res":res}
    # plot_kwargs = {"quantile_bounds": kwargs.get("quantile_bounds", (5, 95)),
    #                "quantiles": kwargs.get("quantiles", [0.05, 0.5, 0.95]),
    #                "verbose": kwargs.get("verbose", True),
    #                "k_true": kwargs.get("k_true", None)}

    _plot(samples, ndim, **plot_kwargs)


def plot_harmonics(coefficients):
    """
    """
    ax:Axes3D
    max_harmonic_degree = int(len(coefficients) ** 0.5 - 1)
    l_vals = np.array([l for l in range(max_harmonic_degree + 1) for _ in range(l+1)])
    m_vals = np.array([m for l in range(max_harmonic_degree + 1) for m in range(l+1)])
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    phi = np.linspace(0, np.pi, 180)
    theta = np.linspace(0, 2*np.pi, 180)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    fcolors = np.zeros((180,180))
    i = 0
    for j in range(len(m_vals)):
        if m_vals[j] == 0:
            fcolors += -coefficients[i]*sph_harm(m_vals[j], l_vals[j], theta, phi).real
            i+=1
        else:
            fcolors += -2*coefficients[i]*sph_harm(m_vals[j], l_vals[j], theta, phi).real
            i+=2
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
    ax.set_axis_off()
    plt.show()





"""
input_dim = 2
hidden_dim = 64

vector_field = SpectralODEVF(input_dim, hidden_dim)
cnf_model = CNF(vector_field)

optimizer = optim.Adam(cnf_model.parameters(), lr=1e-4)

batch_size = 128
num_epochs = 10000

base_dist = GeneralDistribution(distribution_type="uniform", input_dim=input_dim)


def target_distribution(x):
    return torch.distributions.Normal(0, 1).log_prob(x).sum(dim=1)

x0_samples = base_dist.sample(1000)
x1_samples = torch.randn(1000, input_dim)

# Create DataLoader
dataset = TensorDataset(x0_samples, x1_samples)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
min_lr = 1e-8  # Minimum learning rate before stopping


batch_size=128
for epoch in range(num_epochs):
    optimizer.zero_grad()
    x0 = (2*base_dist.sample(batch_size)-1)*2
    x1 = torch.randn(batch_size, input_dim)
    
    l2_lambda = 1e-5
    l2_reg = sum(param.pow(2.0).sum() for param in cnf_model.parameters()) 
    loss = fml_kde(x0, x1, cnf_model.vector_field) #+ l2_lambda * l2_reg
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if optimizer.param_groups[0]['lr'] < min_lr:  # If learning rate is too small, stop training
        print("Stopping early due to minimal learning rate.")
        break
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        z = cnf_model(x0).detach().numpy()
        visualize_samples(x0.detach().numpy(), z, f'Transformed Distribution (Epoch {epoch+1})')
        # visualize_samples(x0.detach().numpy(), x1.detach().numpy(), f'Target Distribution (Epoch {epoch+1})')



num_samples = 1000
base_samples_tensor = 2*(2*base_dist.sample(num_samples)-1)

# Simulate the flow transformation at different time steps
time_steps = torch.linspace(0, 1, steps=15)  # Different stages of the flow
transformed_samples = []

for t in time_steps:
    with torch.no_grad():
        # Assuming `cnf_model` is your trained CNF
        transformed = odeint(cnf_model.vector_field, base_samples_tensor, t.unsqueeze(0), method='rk4')
        transformed_samples.append(transformed[-1].numpy())

# Plot the transformations
fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 6))
print(transformed_samples[0].shape)
for i, t in enumerate(time_steps):
    kde = gaussian_kde(transformed_samples[i].T, bw_method='scott')
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    xgrid = np.linspace(xmin, xmax, 100)
    ygrid = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    axes[i].contourf(X, Y, Z, levels=50, cmap="viridis")
    # axes[i].scatter(transformed_samples[i][:, 0], transformed_samples[i][:, 1], alpha=1)
    axes[i].set_title(f'Time: {t.item():.2f}')
    axes[i].set_xlabel('x1')
    axes[i].set_ylabel('x2')
    axes[i].grid(True)
    axes[i].set_aspect('equal')

plt.suptitle('Transformation of Base Distribution Over Time')
plt.show()

"""