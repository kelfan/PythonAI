class Mediator:
    def Send(self, message, col):
        pass


class Colleague:
    def __init__(self, temp):
        self.mediator = temp

    def Send(self, message):
        self.mediator.Send(message, self)


class Colleague1(Colleague):
    def Notify(self, message):
        print("Colleague1 get A message:%s" % message)


class Colleague2(Colleague):
    def Notify(self, message):
        print("Colleague2 get A message:%s" % message)


# 作为路由器指示信息的走向
class ConcreteMediator(Mediator):
    def Send(self, message, col):
        if col == col1:
            col2.Notify(message)
        else:
            col1.Notify(message)


if __name__ == "__main__":
    m = ConcreteMediator()
    col1 = Colleague1(m)
    col2 = Colleague2(m)
    m.col1 = col1
    m.col2 = col2
    col1.Send("How are you?")
    col2.Send("Fine.")
