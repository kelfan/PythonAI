from abc import abstractmethod


class A:
    @abstractmethod
    def t1(self):
        pass

    @staticmethod
    @abstractmethod
    def t3():
        pass


class C:
    @staticmethod
    def t4():
        print("t4")


class B(A, C):
    def t1(self):
        print("t1")

    def t2(self):
        print("t2")

    @staticmethod
    def t3():
        print("t3")


B().t3()
B().t4()
