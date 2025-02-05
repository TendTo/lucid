from pylucid import Calculator, __version__


def main():
    print("Hello world!")
    print(f"Version: {__version__}")

    c = Calculator()
    a = int(input("Enter a number: "))
    b = int(input("Enter another number: "))
    print(f"{a} + {b} = {c.add(a, b)}")
    print(f"{a} - {b} = {c.subtract(a, b)}")
    print(f"{a} * {b} = {c.multiply(a, b)}")
    print(f"{a} / {b} = {c.divide(a, b)}")


if __name__ == "__main__":
    main()
