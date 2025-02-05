from pylucid import Calculator


class TestFormula:
    def test_constructor(self):
        c = Calculator()
        assert c is not None

    def test_add(self):
        c = Calculator()
        assert c.add(1, 2) == 3

    def test_subtract(self):
        c = Calculator()
        assert c.subtract(1, 2) == -1

    def test_multiply(self):
        c = Calculator()
        assert c.multiply(1, 2) == 2

    def test_divide(self):
        c = Calculator()
        assert c.divide(3, 2) == 3 // 2
