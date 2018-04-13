class Originator:
    def __init__(self):
        self.state = ""

    def Show(self):
        print(self.state)

    def CreateMemo(self):
        return Memo(self.state)

    def SetMemo(self, memo):
        self.state = memo.state


class Memo:
    state = ""

    def __init__(self, ts):
        self.state = ts


class Caretaker:
    memo = ""

# memo 是记录信息的对象
# 类中加入 memo对象的创建和返回
# caretaker 是中间对象存储 memo对象

if __name__ == "__main__":
    on = Originator()
    on.state = "on"
    on.Show()
    c = Caretaker()
    c.memo = on.CreateMemo()
    on.state = "off"
    on.Show()
    on.SetMemo(c.memo)
    on.Show()
