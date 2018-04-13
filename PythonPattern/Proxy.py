class Interface:
    def Request(self):
        return 0


class RealSubject(Interface):
    def Request(self):
        print("Real request.")


# 继承相同interface
# 装入真正物体,然后用装入的输出
class Proxy(Interface):
    def Request(self):
        self.real = RealSubject()
        self.real.Request()


if __name__ == "__main__":
    p = Proxy()
    p.Request()
