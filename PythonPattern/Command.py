class Barbucer:
    def MakeMutton(self):
        print("Mutton")

    def MakeChickenWing(self):
        print("Chicken Wing")


class Command:
    def __init__(self, temp):
        self.receiver = temp

    def ExecuteCmd(self):
        pass


class BakeMuttonCmd(Command):
    def ExecuteCmd(self):
        self.receiver.MakeMutton()


class ChickenWingCmd(Command):
    def ExecuteCmd(self):
        self.receiver.MakeChickenWing()


class Waiter:
    def __init__(self):
        self.order = []

    def SetCmd(self, command):
        self.order.append(command)
        print("Add Order")

    def Notify(self):
        for cmd in self.order:
            # self.order.remove(cmd)
            # lead to A bug
            cmd.ExecuteCmd()


# 动态添加命令
if __name__ == "__main__":
    barbucer = Barbucer()
    cmd = BakeMuttonCmd(barbucer)
    cmd2 = ChickenWingCmd(barbucer)
    girl = Waiter()
    girl.SetCmd(cmd)
    girl.SetCmd(cmd2)
    girl.Notify()

    # remove bug
    # for in 循环中不能对数组进行操作
    c = [0, 1, 2, 3]
    for i in c:
        print(i)
        c.remove(i)

    # output:
    # 0
    # 2
