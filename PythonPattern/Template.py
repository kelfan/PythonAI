class TestPaper:
    def TestQuestion1(self):
        print("Test1:A. B. C. D.")
        print("(%s)" % self.Answer1())

    def TestQuestion2(self):
        print("Test1:A. B. C. D.")
        print("(%s)" % self.Answer2())

    def Answer1(self):
        return ""

    def Answer2(self):
        return ""


class TestPaperA(TestPaper):
    def Answer1(self):
        return "B"

    def Answer2(self):
        return "C"


class TestPaperB(TestPaper):
    def Answer1(self):
        return "D"

    def Answer2(self):
        return "D"

# 定义一个通用的输出,然后定义不同的输出为空方法
# 不同继承类 定义不同的实际方法
if __name__ == "__main__":
    s1 = TestPaperA()
    s2 = TestPaperB()
    print("student 1")
    s1.TestQuestion1()
    s1.TestQuestion2()
    print("student 2")
    s2.TestQuestion1()
    s2.TestQuestion2()
