def printdebug(func):
    def __decorator():
        print('enter the login')
        func()
        print('exit the login')

    return __decorator


@printdebug  # combine the printdebug and login
def login():
    print('in login')


login()  # make the calling point more intuitive


def test2(func):
    def __decorator():
        print("test start")
        func()
        print("test end")

    return __decorator

@printdebug
@test2
def login2():
    print("process login")

login2()
