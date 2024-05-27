import re

class Block:
    def __init__(self, name):
        self.name = name
        self.stmt = []
        self.methods = {}
        self.classes= {}

    def parse(self, fh, prefix=""):
        deco=[]
        for line in fh:
            if res:=re.match(r"\s+@(\w+)", line):
                deco.append(line)
            if res:=re.match(r"\s+class\s+(\w+)", line):
                self.classes[res.group(1)] = Class(res.group(1)).parse(fh)
            elif res:=re.match(r"\s+def\s+(\w+)", line):
                self.methods[res.group(1)] = Function(res.group(1)).parse(fh)
            else:
                self.stmt.append(line)
        return self

    def render(self):
        res = []
        for line in self.stmt:
            res.append(line)
        for _, method in sorted(self.methods.items()):
            res.append(method.render())
        for _, cls in sorted(self.classes.items()):
            res.append(cls.render())
        return "".join(res)

class Class(Block):
    pass

class Function(Block):
    pass

class Source(Block):
    @classmethod
    def process(cls, fh):
        return cls("source").parse(open(fh)).render()