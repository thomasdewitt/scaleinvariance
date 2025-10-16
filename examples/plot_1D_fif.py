#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import scaleinvariance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=float, default=0)
    parser.add_argument('--C1', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--causal', type=bool, default=True)
    parser.add_argument('--size', type=int, default=2**15)
    args = parser.parse_args()

    fif = scaleinvariance.FIF_1D(args.size, args.alpha, args.C1, args.H, causal=args.causal)
    print(fif.mean())

    plt.figure(figsize=(10, 6))
    plt.plot(fif, 'k-', linewidth=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()