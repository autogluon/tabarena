import os

# Your dataset list (converted to Python list of strings)
datasets_string = "190411 359983 189354 189356 10090 359979 168868 190412 146818 359982 359967 359955 359960 359973 359968 359992 359959 359957 359977 7593 168757 211986 168909 189355 359964 359954 168910 359976 359969 359970 189922 359988 359984 360114 359966 211979 168911 359981 359962 360975 3945 360112 359991 359965 190392 359961 359953 359990 359980 167120 359993 190137 359958 190410 359971 168350 360113 359956 359989 359986 359975 359963 359994 359987 168784 359972 190146 359985 146820 359974 2073 359944 359929 233212 359937 359950 359938 233213 359942 233211 359936 359952 359951 359949 233215 360945 167210 359943 359941 359946 360933 360932 359930 233214 359948 359931 359932 359933 359934 359939 359945 359935 317614 359940"
datasets_list = list(map(int, datasets_string.split()))

datasets_regression_string = "359944 359929 233212 359937 359950 359938 233213 359942 233211 359936 359952 359951 359949 233215 360945 167210 359943 359941 359946 360933 360932 359930 233214 359948 359931 359932 359933 359934 359939 359945 359935 317614 359940"
datasets_regression_list = list(map(int, datasets_regression_string.split()))
print(len(datasets_regression_list))

# Get all files in the current directory (or change path as needed)
files = os.listdir("../Metadata/core/core_submatrices")

# Extract dataset numbers from filenames
existing_ids = []
for f in files:
    if f.startswith("Operator_Model_Feature_Matrix_Core") and f.endswith(".parquet"):
        try:
            num = int(f.split("Core")[-1].split(".")[0])
            existing_ids.append(num)
        except ValueError:
            continue

# Find missing IDs
#missing_ids = [ds_id for ds_id in datasets_list if ds_id not in existing_ids]
missing_ids = [(index, ds_id) for index, ds_id in enumerate(datasets_list) if ds_id not in existing_ids]

# Output the result
print("Missing " + str(len(missing_ids)) + " dataset files:")
for index, ds_id in missing_ids:
    print(f"Index: {index}, Dataset ID: {ds_id}")
