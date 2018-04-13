class Person:
    def CreateHead(self):
        pass

    def CreateHand(self):
        pass

    def CreateBody(self):
        pass

    def CreateFoot(self):
        pass


class ThinPerson(Person):
    def CreateHead(self):
        print("thin head")

    def CreateHand(self):
        print("thin hand")

    def CreateBody(self):
        print("thin body")

    def CreateFoot(self):
        print("thin foot")


class ThickPerson(Person):
    def CreateHead(self):
        print("thick head")

    def CreateHand(self):
        print("thick hand")

    def CreateBody(self):
        print("thick body")

    def CreateFoot(self):
        print("thick foot")


class Director:
    def __init__(self, temp):
        self.p = temp

    def Create(self):
        self.p.CreateHead()
        self.p.CreateBody()
        self.p.CreateHand()
        self.p.CreateFoot()


if __name__ == "__main__":
    p = ThickPerson()
    d = Director(p)
    d.Create()
