#!/usr/bin/env python3
"""
3D FIF multifractal visualization with controllable parameters and multiple views.

Generates a 3D fractionally integrated flux field and visualizes it through:
1. Horizontal slice at domain center
2. Vertical slice at domain center
3. 3D isosurface of the 95th percentile as a mostly opaque contour

Supports both isotropic and anisotropic (canonical GSI) scale metrics.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scaleinvariance
from scaleinvariance.simulation.generalized_scale_invariance import canonical_scale_metric


def create_3d_contour(data, level, ax, alpha=0.7, color='cyan'):
    """
    Create a 3D isosurface visualization using marching cubes.

    Parameters
    ----------
    data : ndarray
        3D field
    level : float
        Threshold value for isosurface
    ax : Axes3D
        3D matplotlib axes
    alpha : float
        Transparency of the surface
    color : str
        Color of the surface
    """
    try:
        from skimage import measure

        # Use marching cubes to generate isosurface
        result = measure.marching_cubes(data, level=level, step_size=1)
        vertices = result[0]
        faces = result[1]

        # Create mesh
        mesh = Poly3DCollection(vertices[faces], alpha=alpha, facecolors=color, edgecolors='darkgray', linewidths=0.1)
        ax.add_collection3d(mesh)

    except ImportError:
        # Fallback: use voxels for visualization
        binary = data >= level
        ax.voxels(binary, facecolors=color, alpha=alpha, edgecolors='gray', linewidth=0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 3D FIF field with horizontal/vertical slices and 3D contour."
    )
    parser.add_argument('--H', type=float, default=0.3,
                       help='Hurst exponent (0 <= H <= 1).')
    parser.add_argument('--C1', type=float, default=0.05,
                       help='Codimension of the mean (> 0 for multifractal behavior).')
    parser.add_argument('--alpha', type=float, default=1.8,
                       help='Lévy stability parameter (0 < alpha <= 2, alpha != 1).')
    parser.add_argument('--size', type=int, default=128,
                       help='Size of the 3D domain (creates size x size x size cube).')
    parser.add_argument('--anisotropic', action='store_true',
                       help='Use canonical anisotropic metric (GSI) instead of isotropic.')
    parser.add_argument('--spheroscale', type=float, default=8.0,
                       help='Spheroscale for anisotropic metric (only if --anisotropic).')
    parser.add_argument('--Hz', type=float, default=5/9,
                       help='Vertical anisotropy exponent for canonical metric (only if --anisotropic).')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility.')
    parser.add_argument('--log', action='store_true',
                       help='Plot log10 of the field.')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    size = (args.size, args.size, args.size)

    print(f"Generating 3D FIF: H={args.H}, C1={args.C1}, alpha={args.alpha}, size={size}")

    # Prepare scale metric if anisotropic
    scale_metric = None
    scale_metric_dim = None
    if args.anisotropic:
        print(f"  Using canonical anisotropic metric: spheroscale={args.spheroscale}, Hz={args.Hz}")
        # For non-periodic FIF, the simulation domain is doubled
        scale_metric = canonical_scale_metric(size, ls=args.spheroscale, Hz=args.Hz)
        
        scale_metric_dim = 23/9

    # Generate 3D FIF
    field = scaleinvariance.FIF_ND(
        size=size,
        alpha=args.alpha,
        C1=args.C1,
        H=args.H,
        periodic=True,
        scale_metric=scale_metric,
        scale_metric_dim=scale_metric_dim
    )

    print(f"Field shape: {field.shape}")
    print(f"Field min: {field.min():.6f}, max: {field.max():.6f}, mean: {field.mean():.6f}")

    # Prepare data for plotting
    plot_field = field.copy()
    if args.log:
        positive = plot_field[plot_field > 0]
        if positive.size == 0:
            raise ValueError('Cannot take log of non-positive field values.')
        min_positive = positive.min()
        plot_field = np.log10(np.maximum(plot_field, min_positive))

    # Extract slices at domain center
    # Field indexing convention: [x, y, z] where x,y are horizontal, z is vertical
    center_x = args.size // 2
    center_y = args.size // 2
    center_z = args.size // 2

    # Horizontal slice at constant z (x-y plane), transposed for intuitive plotting
    horizontal_slice = plot_field[:, :, center_z].T    # shape (y, x) for imshow

    # Vertical slice at constant y (x-z plane), transposed for intuitive plotting
    vertical_slice = plot_field[:, center_y, :].T      # shape (z, x) for imshow

    # Calculate 95th percentile for contour
    percentile_95 = np.percentile(plot_field, 95)

    # Create figure with three subplots
    fig = plt.figure(figsize=(12, 4))

    # Subplot 1: Horizontal slice (x-y plane at constant z)
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(horizontal_slice, cmap='magma', aspect='equal', origin='lower')
    ax1.set_title(f'Horizontal Slice (z={center_z})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label('Value', rotation=270, labelpad=15)

    # Subplot 2: Vertical slice (x-z plane at constant y)
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(vertical_slice, cmap='magma', aspect='equal', origin='lower')
    ax2.set_title(f'Vertical Slice (y={center_y})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label('Value', rotation=270, labelpad=15)

    # Subplot 3: 3D isosurface of 95th percentile
    ax3 = fig.add_subplot(133, projection='3d')
    create_3d_contour(plot_field, percentile_95, ax3, alpha=0.6, color='cyan')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'3D Isosurface (95th percentile={percentile_95:.2f})')
    ax3.view_init(elev=20, azim=45)

    # Set consistent axis limits for 3D plot
    ax3.set_xlim(0, args.size)
    ax3.set_ylim(0, args.size)
    ax3.set_zlim(0, args.size)

    # Main title
    title = f"3D FIF: H={args.H}, C1={args.C1}, alpha={args.alpha}"
    if args.anisotropic:
        title += f" (Anisotropic: Hz={args.Hz}, ls={args.spheroscale})"
    else:
        title += " (Isotropic)"
    if args.log:
        title += " (log10)"
    fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
