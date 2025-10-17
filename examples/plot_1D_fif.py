#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import scaleinvariance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=float, default=0)
    parser.add_argument('--C1', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--causal', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--size', type=int, default=2**20)
    args = parser.parse_args()
    print(f'H: {args.H} C1: {args.C1} alpha: {args.alpha} Causal: {args.causal}')

    fif = scaleinvariance.FIF_1D(args.size, args.alpha, args.C1, args.H, causal=args.causal)

    plt.figure(figsize=(10, 6))
    plt.plot(fif, 'k-', linewidth=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()