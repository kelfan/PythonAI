class SubSystemOne:
    def MethodOne(self):
        print("SubSysOne")


class SubSystemTwo:
    def MethodTwo(self):
        print("SubSysTwo")


class SubSystemThree:
    def MethodThree(self):
        print("SubSysThree")


class SubSystemFour:
    def MethodFour(self):
        print("SubSysFour")


class Facade:
    def __init__(self):
        self.one = SubSystemOne()
        self.two = SubSystemTwo()
        self.three = SubSystemThree()
        self.four = SubSystemFour()

    def MethodA(self):
        print("MethodA")
        self.one.MethodOne()
        self.two.MethodTwo()
        self.four.MethodFour()

    def MethodB(self):
        print("MethodB")
        self.two.MethodTwo()
        self.three.MethodThree()


if __name__ == "__main__":
    facade = Facade()
    facade.MethodA()
    facade.MethodB()
