# -*- coding: UTF-8 -*-
class Person:
    def Accept(self, visitor):
        pass


class Man(Person):
    def Accept(self, visitor):
        visitor.GetManConclusion(self)


class Woman(Person):
    def Accept(self, visitor):
        visitor.GetWomanConclusion(self)


class Action:
    def GetManConclusion(self, concreteElementA):
        pass

    def GetWomanConclusion(self, concreteElementB):
        pass


class Success(Action):
    def GetManConclusion(self, concreteElementA):
        print("男人成功时，背后有个伟大的女人")

    def GetWomanConclusion(self, concreteElementB):
        print("女人成功时，背后有个不成功的男人")


class Failure(Action):
    def GetManConclusion(self, concreteElementA):
        print("男人失败时，闷头喝酒，谁也不用劝")

    def GetWomanConclusion(self, concreteElementB):
        print("女人失败时，眼泪汪汪，谁也劝不了")


class ObjectStructure:
    def __init__(self):
        self.plist = []

    def Add(self, p):
        self.plist = self.plist + [p]

    def Display(self, act):
        for p in self.plist:
            p.Accept(act)

# 可以 多个对象实现一步顺序输出
# 对象中定义使用的方法
# 输出类中定义各种方法的输出
if __name__ == "__main__":
    os = ObjectStructure()
    os.Add(Man())
    os.Add(Woman())
    os.Add(Man())
    os.Add(Man())
    sc = Success()
    os.Display(sc)
    fl = Failure()
    os.Display(fl)
