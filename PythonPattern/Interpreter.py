class Context:
    def __init__(self):
        self.input = ""
        self.output = ""


class AbstractExpression:
    def Interpret(self, context):
        pass


class Expression(AbstractExpression):
    def Interpret(self, context):
        print("terminal interpret")


class NonterminalExpression(AbstractExpression):
    def Interpret(self, context):
        print("Nonterminal interpret")

# 内容用不同的对象去解析
if __name__ == "__main__":
    context = ""
    c = []
    c = c + [Expression()]
    c = c + [NonterminalExpression()]
    c = c + [Expression()]
    c = c + [Expression()]
    # c 是 对象数组
    for a in c:
        a.Interpret(context)
