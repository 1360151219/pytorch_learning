# Data Preprocessing — 数据预处理
import os
import pandas as pd


def make_file():
    os.makedirs("dataset", exist_ok=True)
    data_file = os.path.join("dataset", "index.csv")
    with open(data_file, "w") as f:
        f.write("NumRooms,Alley,Price\n")
        f.write("NA,Pave,10000\n")
        f.write("2,NA,20000\n")
        f.write("4,NA,30000\n")
        f.write("NA,NA,40000\n")
        f.close()


def main():
    # make_file()
    data_file = os.path.join("dataset", "index.csv")
    data = pd.read_csv(data_file)

    #    NumRooms Alley  Price
    # 0       NaN  Pave  10000
    # 1       2.0   NaN  20000
    # 2       4.0   NaN  30000
    # 3       NaN   NaN  40000

    # index location 取所有行中的0-1列
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

    # 分别处理数值列和字符串列
    numeric_cols = inputs.select_dtypes(include="number").columns
    # 数值列：用均值填充。第一列中第0、3行是空，因此填充一个均值（(2+4)/2=3.0）
    inputs[numeric_cols] = inputs[numeric_cols].fillna(inputs[numeric_cols].mean())

    #        NumRooms Alley
    # 0       3.0  Pave
    # 1       2.0   NaN
    # 2       4.0   NaN
    # 3       3.0   NaN
    inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
    print(inputs, inputs.values)


if __name__ == "__main__":
    main()
