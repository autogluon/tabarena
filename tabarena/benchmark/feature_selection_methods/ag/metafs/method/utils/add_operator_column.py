import pandas as pd


def main():
    path = "../Operator_Model_Feature_Matrix_2_168911.parquet"
    matrix = pd.read_parquet(path, engine='pyarrow')

    featurename = pd.Series(matrix["feature - name"])
    operator = featurename.apply(lambda x: x.split('(')[0])
    matrix.insert(4, "operator", operator)
    pd.set_option('display.max_columns', None)
    print(matrix)
    matrix.to_parquet(path)


if __name__ == '__main__':
    main()
