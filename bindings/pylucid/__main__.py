from pylucid import GaussianKernel, Kernel, __version__


def main():
    print("Hello world!")
    print(f"Version: {__version__}")
    print(GaussianKernel)
    print(GaussianKernel([1, 2, 3, 4]))
    print(GaussianKernel([1, 2, 3, 4]).clone())
    g = GaussianKernel([1, 2, 3, 4])
    print(g is g)
    print(Kernel)
    print(g is g.clone())
    print(g([1, 2, 3], [2, 4, 6]))
    print(g([1, 2, 3], [2, 4, 6]))


if __name__ == "__main__":
    main()
