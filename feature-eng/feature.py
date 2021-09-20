from itertools import permutations
from typing import List

binary_ops = ["+", "*", "/"]
unary_ops = ["^0.5", "^3"]


class Operator:
    def __init__(self, operator: str):
        assert operator in binary_ops or operator in unary_ops
        self.operator = operator.strip()

    def __repr__(self):
        return self.operator

    def is_unary(self):
        return self.operator in unary_ops

    def exec(self, v1, v2):
        if self.operator == "+":
            return v1 + v2
        if self.operator == "-":
            return v1 - v2
        if self.operator == "*":
            return v1 * v2
        if self.operator == "/":
            return v1 / v2
        if self.operator == "^0.5":
            return v1 ** 0.5
        if self.operator == "^0.75":
            return v1 ** 0.75
        if self.operator == "^2":
            return v1 ** 2
        if self.operator == "^3":
            return v1 ** 3
        if self.operator == "^4":
            return v1 ** 4

    def wrap_exec(self, v1, v2):
        val = exec(v1, v2)
        f = Feature("VALUE")
        f.value = val
        return f

    def operate(self, p1, p2):
        assert type(p1) == Feature
        if self.is_unary():
            return self.wrap_exec(p1.value, p2)
        else:
            return self.wrap_exec(p1.value, p2.value)


class Feature:
    def __init__(self, name: str):
        self.name = name
        self.value = None

    def __repr__(self):
        return self.name


class FeatureCross:
    def __repr__(self):
        return str(self.f1) + str(self.symbol) + str(self.f2)

    def __init__(self, f1, f2, symbol: str):
        self.f1 = f1
        self.f2 = f2
        self.symbol = symbol
        self.cross = []
        if type(f1) == Feature:
            self.cross.append(f1)
        elif type(f1) == FeatureCross:
            self.cross.extend(f1.cross)
        else:
            raise Exception("Undefined feature cross")
        self.cross.append(symbol)
        if f2:
            if type(f2) == Feature:
                self.cross.append(f2)
            elif type(f2) == FeatureCross:
                self.cross.extend(f2.cross)
            elif f2 is None:
                pass
            else:
                raise Exception("Undefined feature cross")

    def execute_cross(self) -> float:
        assert len(self.cross) > 0
        while self.cross[-1].name != "VALUE":
            param1 = self.cross.pop(0)
            param2 = self.cross.pop(0)
            assert type(param1) == Feature and type(param2) == Operator
            if param2.is_unary():
                out = param2.operate(param1, None)
                self.cross = [out] + self.cross
            else:
                param3 = self.cross.pop(0)
                out = param2.operate(param1, param3)
                self.cross = [out] + self.cross
        assert len(self.cross) == 1
        return self.cross[0].value


def get_permutes(lst: List, n):
    out = []
    if n == 3:
        for item in permutations(range(len(lst)), 3):
            if type(lst[item[1]]) == Operator and ((type(lst[item[0]]) == Feature or type(lst[item[0]]) == FeatureCross)
                                                   and (type(lst[item[2]]) == Feature or type(lst[item[2]]) ==
                                                        FeatureCross)):
                if not lst[item[1]].is_unary():
                    out.append((lst[item[0]], lst[item[1]], lst[item[2]]))
        return out
    elif n == 2:
        for item in permutations(range(len(lst)), 2):
            if type(lst[item[1]]) == Operator and lst[item[1]].is_unary() and (
                    type(lst[item[0]]) == Feature or type(lst[item[0]]) == FeatureCross):
                out.append((lst[item[0]], lst[item[1]]))
        return out
    else:
        raise Exception("Not allowed more than 3 permutes")


if __name__ == '__main__':
    features = ['a', 'b', 'c', 'd']
    ops = binary_ops + unary_ops
    all_val = [Feature(i) for i in features] + [Operator(i) for i in ops]
    crosses_l1 = []
    all_val.extend([FeatureCross(cr[0], cr[2], cr[1]) for cr in get_permutes(all_val, 3)])
    # print(all_val)
    all_val.extend(get_permutes(all_val, 3))
    print(all_val)
