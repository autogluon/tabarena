def get_operators():
    # Unary OpenFE Operators
    unary_operators = [
        "min",
        "max",
        "freq",
        "abs",
        "log",
        "sqrt",
        "square",
        "sigmoid",
        "round",
        "residual"
    ]

    # Binary OpenFE Operators
    binary_operators = [
        "+",
        "-",
        "*",
        "/",
        "GroupByThenMin",
        "GroupByThenMax",
        "GroupByThenMean",
        "GroupByThenMedian",
        "GroupByThenStd",
        "GroupByThenRank",
        "Combine",
        "CombineThenFreq",
        "GroupByThenNUnique"
    ]
    return unary_operators, binary_operators
