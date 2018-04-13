class LeiFeng:
    def Sweep(self):
        print("LeiFeng sweep")


class Student(LeiFeng):
    def Sweep(self):
        print("Student sweep")


class Volenter(LeiFeng):
    def Sweep(self):
        print("Volenter sweep")


class LeiFengFactory:
    def CreateLeiFeng(self):
        temp = LeiFeng()
        return temp


class StudentFactory(LeiFengFactory):
    def CreateLeiFeng(self):
        temp = Student()
        return temp


class VolenterFactory(LeiFengFactory):
    def CreateLeiFeng(self):
        temp = Volenter()
        return temp

# 一个工厂对应一个类的实例化
if __name__ == "__main__":
    sf = StudentFactory()
    s = sf.CreateLeiFeng()
    s.Sweep()
    sdf = VolenterFactory()
    sd = sdf.CreateLeiFeng()
    sd.Sweep()
