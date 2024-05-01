from .utils import print_diff_dict_float

class Knob:
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["value"], data["weights"])

    @classmethod
    def from_src(cls, src):
        if  hasattr(src, "name") and hasattr(src, "value") and hasattr(src, "weights"):
            return cls(src.name, src.value, src.weights)
        else:
            return cls.from_dict(src)

    def __init__(self, name, value=0, weights=None):
        self.name = name
        self.value = value
        self.weights = weights

    def from_madx(self, madx, redefine_weights=False):
        if weights is None:
            weights = {}

    def __repr__(self):
        return f"Knob({self.name!r}, {self.value})"

    def copy(self):
        return Knob(self.name, self.value, self.weights.copy())

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "weights": self.weights,
        }

    def diff(self, other):
        if self.value != other.value:
            print(f"{self.name} {self.value} != {other.value}")
        print_diff_dict_float(self.weights, other.weights)
