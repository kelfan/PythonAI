from Tools.scripts.treesync import raw_input


class Operation:
    def GetResult(self):
        pass


class OperationAdd(Operation):
    def GetResult(self):
        return self.op1 + self.op2


class OperationSub(Operation):
    def GetResult(self):
        return self.op1 - self.op2


class OperationMul(Operation):
    def GetResult(self):
        return self.op1 * self.op2


class OperationDiv(Operation):
    def GetResult(self):
        try:
            result = self.op1 / self.op2
            return result
        except:
            print("error:divided by zero.")
            return 0


class OperationUndef(Operation):
    def GetResult(self):
        print("Undefine operation.")
        return 0


# 关键是这里,根据不同的Key获取对应的方法
class OperationFactory:
    operation = {"+": OperationAdd(), "-": OperationSub(), "*": OperationMul(), "/": OperationDiv()}

    def createOperation(self, ch):
        if ch in self.operation:
            op = self.operation[ch]
        else:
            op = OperationUndef()
        return op


if __name__ == "__main__":
    op = "+"
    opa = 14
    opb = 15
    factory = OperationFactory()
    cal = factory.createOperation(op)
    cal.op1 = opa
    cal.op2 = opb
    print(cal.GetResult())
