#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

import scaleinvariance


def main():
    parser = argparse.ArgumentParser(description="Visualize a 2D FIF field using magma colormap.")
    parser.add_argument('--H', type=float, default=0.3, help='Hurst exponent (0 <= H <= 1).')
    parser.add_argument('--C1', type=float, default=0.05, help='Codimension of the mean (> 0 for multifractal behavior).')
    parser.add_argument('--alpha', type=float, default=1.8, help='Lévy stability parameter (0 < alpha <= 2, alpha != 1).')
    parser.add_argument('--size', type=int, default=2048, help='Output height of the simulated field (even number).')
    parser.add_argument('--width', type=int, default=None, help='Output width of the field. Defaults to --size if omitted.')
    parser.add_argument('--periodic', action='store_true', help='Generate strictly periodic field without internal padding.')
    parser.add_argument('--log', action='store_true', help='Plot log10 of the field to emphasize multifractal contrast.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()

    if args.width is None:
        width = args.size
    else:
        width = args.width

    if args.seed is not None:
        np.random.seed(args.seed)

    field = scaleinvariance.FIF_2D(
        size=(args.size, width),
        alpha=args.alpha,
        C1=args.C1,
        H=args.H,
        periodic=args.periodic,
    )

    print(f"Field mean: {field.mean():.6f}")
    print(f"Field shape: {field.shape}")

    plot_field = field.copy()
    if args.log:
        positive = plot_field[plot_field > 0]
        if positive.size == 0:
            raise ValueError('Cannot take log of non-positive field values.')
        min_positive = positive.min()
        plot_field = np.log10(np.maximum(plot_field, min_positive))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(plot_field, cmap='magma', aspect='equal', origin='lower')
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Field value', rotation=270, labelpad=15)

    title = f"FIF_2D: H={args.H}, C1={args.C1}, alpha={args.alpha}"
    if args.periodic:
        title += " (periodic)"
    if args.log:
        title += " (log10)"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
