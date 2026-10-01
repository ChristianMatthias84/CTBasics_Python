import numpy as np


def main():
    x = np.linspace(0.0, 1.0, 8)
    return x.sum()


if __name__ == '__main__':
    print(main())