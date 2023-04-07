import pandas as pd
import matplotlib.pyplot as plt

# 数据清理，弄成一个个小表格
data_RF = pd.read_excel("./data.xlsx", sheet_name="RF")
data_GBDT = pd.read_excel("./data.xlsx", sheet_name="GBDT")
data_KNN = pd.read_excel("./data.xlsx", sheet_name="KNN")


def paint(df, num, num1):

    if (num == 1):
        plt.plot(df[0], label='CAE', linewidth=1, color='c', marker='o', markerfacecolor='blue', markersize=7)
        plt.plot(df[1], label='PCA', linewidth=1, color='r', marker='v', markerfacecolor='r', markersize=7)
        plt.plot(df[2], label='SVM', linewidth=1, color='g', marker='s', markerfacecolor='g', markersize=7)
        plt.plot(df[3], label='AE', linewidth=1, color='y', marker='p', markerfacecolor='y', markersize=7)
        plt.ylabel('ACC', fontsize=15)

    if (num == 2):
        plt.plot(df[6], label='CAE', linewidth=1, color='c', marker='o', markerfacecolor='blue', markersize=7)
        plt.plot(df[7], label='PCA', linewidth=1, color='r', marker='v', markerfacecolor='r', markersize=7)
        plt.plot(df[8], label='SVM', linewidth=1, color='g', marker='s', markerfacecolor='g', markersize=7)
        plt.plot(df[9], label='AE', linewidth=1, color='y', marker='p', markerfacecolor='y', markersize=7)
        plt.ylabel('PRE', fontsize=15)

    if (num == 3):
        plt.plot(df[12], label='CAE', linewidth=1, color='c', marker='o', markerfacecolor='blue', markersize=7)
        plt.plot(df[13], label='PCA', linewidth=1, color='r', marker='v', markerfacecolor='r', markersize=7)
        plt.plot(df[14], label='SVM', linewidth=1, color='g', marker='s', markerfacecolor='g', markersize=7)
        plt.plot(df[15], label='AE', linewidth=1, color='y', marker='p', markerfacecolor='y', markersize=7)
        plt.ylabel('REC', fontsize=15)

    if (num == 4):
        plt.plot(df[18], label='CAE', linewidth=1, color='c', marker='o', markerfacecolor='blue', markersize=7)
        plt.plot(df[19], label='PCA', linewidth=1, color='r', marker='v', markerfacecolor='r', markersize=7)
        plt.plot(df[20], label='SVM', linewidth=1, color='g', marker='s', markerfacecolor='g', markersize=7)
        plt.plot(df[21], label='AE', linewidth=1, color='y', marker='p', markerfacecolor='y', markersize=7)
        plt.ylabel('F-Score', fontsize=15)

    plt.xlabel("香型", fontsize=15)
    title(num1)
    # 图例说明
    plt.legend()
    # 显示图像
    plt.show()


def title(title):
    if (title == 1):
        plt.title("RF", fontsize=15)
    if (title == 2):
        plt.title("GBDT", fontsize=15)
    if (title == 3):
        plt.title("KNN", fontsize=15)


def clean(data, num):
    data = data.drop(data.columns[0], axis=1)
    data = data.drop(data.columns[0], axis=1)
    df_1 = data.iloc[:4, :]
    df_2 = data.iloc[6:10, :]
    df_3 = data.iloc[12:16, :]
    df_4 = data.iloc[18:22, :]
    df_1 = df_1.T
    df_2 = df_2.T
    df_3 = df_3.T
    df_4 = df_4.T
    paint(df_1, 1, num)
    paint(df_2, 2, num)
    paint(df_3, 3, num)
    paint(df_4, 4, num)


clean(data_RF, 1)
clean(data_GBDT,2)
clean(data_KNN,3)
