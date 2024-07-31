import re

class Block:
    def __init__(self):
        self.head = None
        self.methods = {}
        self.classes = {}

    def parse(self, fh):
        block=[]
        name=None
        typ=None
        for line in fh:
            isdeco=line.startswith("@")
            isclass=linestartswith("class")
            isdef = linestartswith("def")
            if isdeco or isclass or isdef:
                src="".join(block)
                if name is None:
                    self.head=src
                else:
                    if typ=='def':
                        self.methods[name]=src
                    elif typ=='class':
                        self.classes[name]=src
                block=[]
           if isclass:
               name=re.match(r"class\s+(\w+)",line).group(1)
           elif isdef:
               name=re.match(r"def\s+(\w+)",line).group(1)
           block.append(line)
        return self

    def render(self):
        res = [self.headr]
        for _, method in sorted(self.methods.items()):
            res.append(method)
        for _, cls in sorted(self.classes.items()):
            res.append(cls)
        return "".join(res)

if __name__=="__main__":
    import sys
    fh=open(sys.argv[1])
    Source.parse(fh).render()
