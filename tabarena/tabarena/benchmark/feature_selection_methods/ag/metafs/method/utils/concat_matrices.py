from glob import glob
import pandas as pd

files = glob('../Metadata/SubMatrices')
result_matrix = pd.DataFrame()
for file in files:
    file_matrix = pd.read_parquet(file)
    result_matrix = pd.concat([result_matrix, file_matrix], ignore_index=True)
result_matrix.to_parquet('../Metadata/Operator_Model_Feature_Matrix_2.parquet')

