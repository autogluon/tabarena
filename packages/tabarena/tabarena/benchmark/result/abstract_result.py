import copy


class AbstractResult:
    def __init__(self, result: dict, inplace: bool = False):
        if not inplace:
            result = copy.deepcopy(result)
        self.result: dict = result
