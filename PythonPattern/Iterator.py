# encoding=utf-8
#
# by panda
# 迭代器(Iterator)模式

# 迭代器抽象类
class Iterator:
    def First(self):
        pass

    def Next(self):
        pass

    def IsDone(self):
        pass

    def CurrentItem(self):
        pass

    # 集合抽象类


class Aggregate:
    def CreateIterator(self):
        pass

    # 具体迭代器类：


class ConcreteIterator(Iterator):
    aggregate = None
    current = 0

    def __init__(self, aggregate):
        self.aggregate = aggregate
        self.current = 0

    def First(self):
        return self.aggregate[0]

    def Next(self):
        ret = None
        self.current += 1
        if (self.current < len(self.aggregate)):
            ret = self.aggregate[self.current]
        return ret

    def IsDone(self):
        if (self.current < len(self.aggregate)):
            return False
        else:
            return True

    def CurrentItem(self):
        ret = None
        if (self.current < len(self.aggregate)):
            ret = self.aggregate[self.current]
        return ret

    # 具体集合类


class ConcreteAggregate(Aggregate):
    items = None

    def __init__(self):
        self.items = []


def clientUI():
    a = ConcreteAggregate()
    a.items.append('大鸟')
    a.items.append('小菜')
    a.items.append('行李')
    a.items.append('老外')
    a.items.append('公交内部员工')
    a.items.append('小偷')

    print('---------迭代器模式-------------')
    i = ConcreteIterator(a.items)
    item = i.First()
    while (False == i.IsDone()):
        print("%s 请买车票！" % i.CurrentItem())
        i.Next()

    print('\n---------python内部迭代-------------')
    for item in a.items:
        print("%s 请买车票！" % item)
    return


if __name__ == '__main__':
    clientUI()