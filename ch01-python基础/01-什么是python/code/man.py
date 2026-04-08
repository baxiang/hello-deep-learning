class Man:
    """示例类：描述一个人"""

    def __init__(self, name):
        self.name = name
        print("初始化完成！")

    def hello(self):
        print("你好，我是" + self.name + "！")

    def goodbye(self):
        print("再见，" + self.name + "！")


m = Man("大卫")
m.hello()
m.goodbye()
