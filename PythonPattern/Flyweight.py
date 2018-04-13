import sys


class WebSite:
    def Use(self):
        pass


class ConcreteWebSite(WebSite):
    def __init__(self, strName):
        self.name = strName

    def Use(self, user):
        print("Website type:%s,user:%s" % (self.name, user))


class UnShareWebSite(WebSite):
    def __init__(self, strName):
        self.name = strName

    def Use(self, user):
        print("UnShare Website type:%s,user:%s" % (self.name, user))

# 收集产生的实例
class WebFactory:
    def __init__(self):
        test = ConcreteWebSite("test")
        self.webtype = {"test": test}
        self.count = {"test": 0}

    def GetWeb(self, webtype):
        """
        汇集统计计算
        :param webtype:
        :return:
        """
        if webtype not in self.webtype:
            temp = ConcreteWebSite(webtype)
            self.webtype[webtype] = temp
            self.count[webtype] = 1
        else:
            temp = self.webtype[webtype]
            self.count[webtype] = self.count[webtype] + 1
        return temp

    def GetCount(self):
        for key in self.webtype:
            # print "type: %s, count:%d" %(key,sys.getrefcount(self.webtype[key]))
            print("type: %s, count:%d " % (key, self.count[key]))

# 方便汇集统计计算
if __name__ == "__main__":
    f = WebFactory()
    ws = f.GetWeb("show")
    ws.Use("Lee")
    ws2 = f.GetWeb("show")
    ws2.Use("Jack")
    ws3 = f.GetWeb("blog")
    ws3.Use("Chen")
    ws4 = UnShareWebSite("TEST")
    ws4.Use("Mr.Q")
    print(f.webtype)
    f.GetCount()
