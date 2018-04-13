class Person:
    def __init__(self, tname):
        self.name = tname

    def Show(self):
        print("dressed %s" % (self.name))


class Finery(Person):
    componet = None

    def __init__(self):
        pass

    def Decorate(self, ct):
        self.componet = ct

    def Show(self):
        if (self.componet != None):
            self.componet.Show()


class TShirts(Finery):
    def __init__(self):
        pass

    def Show(self):
        print("Big T-shirt ")
        self.componet.Show()


class BigTrouser(Finery):
    def __init__(self):
        pass

    def Show(self):
        print("Big Trouser ")
        self.componet.Show()


# 适合一层后累加下一层, 输出是最外层的类先输出
# 需要的是finery类中有component装入新的类
# 需要输出自身后输出装入的component的输出
if __name__ == "__main__":
    p = Person("somebody")
    bt = BigTrouser()
    ts = TShirts()
    bt.Decorate(p)
    ts.Decorate(bt)
    ts.Show()
