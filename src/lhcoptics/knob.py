class Knob:
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["value"], data["weights"])

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
