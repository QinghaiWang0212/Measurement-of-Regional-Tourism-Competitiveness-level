import numpy as np
np.set_printoptions(threshold=np.inf)
import genetic_algorithm
import problem
import excelextract
import pandas as pd
import math
import copy


# 将各城市数据横向连接为combined_list,标准化后存入three_D_array
# 由three_D_array求各市历年均值，存入二维矩阵three_D_array_mean，代入entropy_weight_method求权重，和迭代值矩阵相乘得评分

city=["jinan","zibo","taian","liaocheng","dezhou","binzhou","dongying","laiwu"]
city1 = ["jinan", "zibo", "taian", "liaocheng", "dezhou", "binzhou", "dongying", "laiwu"]
city_sheet=["济南","淄博","泰安","聊城","德州","滨州","东营","莱芜"]
for i in range(len(city)):
    city[i]="result_"+city[i]+".xlsx"

# 使用xlrd方法读excel，excelextract.get_list_data_from_excel()要使用rowvalvue：
# # 旅游指标表格中间有断行
# selected_rows=[3,4,5,6,7,9,10,11,12,14,15,16,17]
# # 表格左侧有列标，且前两列数据未收集全，暂不使用
# selected_cols=[i for i in range(3,14+1)]


# 使用openpyxl方法读excel
# 旅游指标表格中间有断行
selected_rows=[4,5,6,7,8,10,11,12,13,15,16,17,18]
# 表格左侧有列标，且前两列数据未收集全，暂不使用
selected_cols=[i for i in range(4,15+1)]
rows = len(selected_rows)
cols = len(selected_cols)-1

# 获取单元格值的两种方法
# cellvalue=sn["C2"].value
# cellvalue=sn.cell(2,3).value

three_D_array = np.zeros((len(city_sheet), rows, cols))
combined_list = []
standard_combined_array=np.zeros((rows,cols*len(city)))
# 将"旅游指标.xlsx"中各市的数据读入二维列表combined_list中，逐个表横向连接
for i in range(len(city_sheet)):
    sheet = excelextract.get_sheet_by_name("旅游指标.xlsx", city_sheet[i])
    original_data = excelextract.get_list_data_from_excel(sheet, selected_rows=selected_rows, selected_cols=selected_cols)
    if not np.array(combined_list).any():
        combined_list = np.array(original_data)[:,:-1].astype(np.float).copy()
    else:
        combined_list = np.column_stack((combined_list,np.array(original_data)[:,:-1])).astype(np.float)


# combined_list增加一列，该列是最后一列，放置了各行是正向指标还是负向指标的标识
combined_list=np.column_stack((combined_list,np.array(original_data)[:,-1]))
# standard_combined_array存放combine_list标准化后的数据
for i in range(len(combined_list)):
    standard_combined_array[i]=excelextract.standardize_extend_interval(combined_list[i])

# standard_combined_array的数据三维化存入three_D_array 中

for i in range(len(city)):
    three_D_array[i]=standard_combined_array[:,i*cols:(i+1)*cols]
    for j in range(len(three_D_array[i])):
        three_D_array[i][j]=three_D_array[i][j][::-1]

# three_D_array结果写入"three_D_array.xlsx"
writer_three_D_array = pd.ExcelWriter("three_D_array.xlsx")
for i in range(len(three_D_array)):
    write_three_D_array = pd.DataFrame(three_D_array[i])
    write_three_D_array.to_excel(writer_three_D_array,"sheet1",float_format="%.3f",startrow=2+i*15)

writer_three_D_array.save()
writer_three_D_array.close()
#
# 将各市标准化值取均值汇总在一个二维矩阵three_D_array_mean中
three_D_array_mean = np.zeros((len(three_D_array[0]),len(three_D_array)))
for i in range(len(three_D_array[0])):
    for j in range(len(three_D_array)):
        three_D_array_mean[i][j] = np.mean(three_D_array[j][i])

# 将各市同年二级指标数据求均值为相应一级指标数据，一级指标数据汇总到first_level_indicator_data 中
# first_level_indicator_data = np.zeros((24,11))
# for i in range(len(rod.three_D_array)):
#     first_level_indicator_data[i*3+0] = np.mean(rod.three_D_array[i][0:5],axis=0)
#     first_level_indicator_data[i*3+1] = np.mean(rod.three_D_array[i][5:9],axis=0)
#     first_level_indicator_data[i*3+2] = np.mean(rod.three_D_array[i][9:13],axis=0)


# second_level_indicator_data存放各城市历年标准化数据，是three_D_array的二维化，等同于standard_combined_array?

second_level_indicator_data = np.zeros((104,11))
for i in range(three_D_array.shape[0]):
    for j in range(three_D_array.shape[1]):
        second_level_indicator_data[i*13+j] = three_D_array[i][j]



entropy_weights = excelextract.entropy_weight_method(three_D_array_mean)

# 读取fcm_weights.xlsx和fcm_iteration_second_indicator.xlsx数据
# sheet1 = excelextract.get_sheet_by_sequence("fcm_weights.xlsx", 0)
# fcm_weights = excelextract.get_array_data_from_excel(sheet1,104,104,2,2)
#
#
#
# sheet2 = excelextract.get_sheet_by_sequence("fcm_iteration_second_indicator.xlsx", 0)
# fcm_iteration = excelextract.get_array_data_from_excel(sheet2,29,104,2,2)


"""
# 将各市二级指标均值用熵权法求权重，和fcm迭代结果乘积求各二级指标评分，然后汇算各市一级指标评分，
# 写入first_indicator_iteration_results_scores.xlsx

iteration_results = fcm_iteration[-1,:]
iteration_results_scores = np.zeros((13,8))
for j in range(iteration_results_scores.shape[1]):
    iteration_results_scores[:,j]=iteration_results[j*13:13+j*13] * entropy_weights
first_indicator_iteration_results_scores= np.zeros((3,8))
for j in range(first_indicator_iteration_results_scores.shape[1]):
    first_indicator_iteration_results_scores[0][j] = iteration_results_scores[0:5, j].sum()
    first_indicator_iteration_results_scores[1][j] = iteration_results_scores[5:9, j].sum()
    first_indicator_iteration_results_scores[2][j] = iteration_results_scores[9:13, j].sum()


writer_first_indicator_iteration_results_scores = pd.ExcelWriter("first_indicator_iteration_results_scores.xlsx")
write_first_indicator_iteration_results_scores = pd.DataFrame(first_indicator_iteration_results_scores)
write_first_indicator_iteration_results_scores.to_excel(writer_first_indicator_iteration_results_scores,"sheet1",float_format="%.3f",startrow=1)
writer_first_indicator_iteration_results_scores.save()

"""
