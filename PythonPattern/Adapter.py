class Target:
    def Request(self):
        print("common request.")


class Adaptee(Target):
    def SpecificRequest(self):
        print("specific request.")

# 转成端口共用的方法
class Adapter(Target):
    def __init__(self, ada):
        self.adaptee = ada

    def Request(self):
        self.adaptee.SpecificRequest()


if __name__ == "__main__":
    adaptee = Adaptee()
    adapter = Adapter(adaptee)
    adapter.Request()
