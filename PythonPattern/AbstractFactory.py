class IUser:
    def GetUser(self):
        pass

    def InsertUser(self):
        pass


class IDepartment:
    def GetDepartment(self):
        pass

    def InsertDepartment(self):
        pass


class CAccessUser(IUser):
    def GetUser(self):
        print("Access GetUser")

    def InsertUser(self):
        print("Access InsertUser")


class CAccessDepartment(IDepartment):
    def GetDepartment(self):
        print("Access GetDepartment")

    def InsertDepartment(self):
        print("Access InsertDepartment")


class CSqlUser(IUser):
    def GetUser(self):
        print("Sql GetUser")

    def InsertUser(self):
        print("Sql InsertUser")


class CSqlDepartment(IDepartment):
    def GetDepartment(self):
        print("Sql GetDepartment")

    def InsertDepartment(self):
        print("Sql InsertDepartment")


class IFactory:
    def CreateUser(self):
        pass

    def CreateDepartment(self):
        pass


class AccessFactory(IFactory):
    def CreateUser(self):
        temp = CAccessUser()
        return temp

    def CreateDepartment(self):
        temp = CAccessDepartment()
        return temp


class SqlFactory(IFactory):
    def CreateUser(self):
        temp = CSqlUser()
        return temp

    def CreateDepartment(self):
        temp = CSqlDepartment()
        return temp


if __name__ == "__main__":
    factory = SqlFactory()
    user = factory.CreateUser()
    depart = factory.CreateDepartment()
    user.GetUser()
    depart.GetDepartment()
