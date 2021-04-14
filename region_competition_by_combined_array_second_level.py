import numpy as np
# 使numpy矩阵信息打印显示完全：
np.set_printoptions(threshold=np.inf)
import openpyxl
import excelextract
import pandas as pd
import copy
import genetic_algorithm
import problem
import  read_original_data as rod





city=["jinan","zibo","taian","liaocheng","dezhou","binzhou","dongying","laiwu"]
city1 = ["jinan", "zibo", "taian", "liaocheng", "dezhou", "binzhou", "dongying", "laiwu"]
city_sheet=["济南","淄博","泰安","聊城","德州","滨州","东营","莱芜"]
for i in range(len(city)):
    city[i]="result_"+city[i]+".xlsx"
rows = 13
cols = 11
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



# 将各市同年二级指标数据求均值为相应一级指标数据，一级指标数据汇总到first_level_indicator_data 中
# first_level_indicator_data = np.zeros((24,11))
# for i in range(len(rod.three_D_array)):
#     first_level_indicator_data[i*3+0] = np.mean(rod.three_D_array[i][0:5],axis=0)
#     first_level_indicator_data[i*3+1] = np.mean(rod.three_D_array[i][5:9],axis=0)
#     first_level_indicator_data[i*3+2] = np.mean(rod.three_D_array[i][9:13],axis=0)

second_level_indicator_data = np.zeros((104,11))
for i in range(rod.three_D_array.shape[0]):
    for j in range(rod.three_D_array.shape[1]):
        second_level_indicator_data[i*13+j] = rod.three_D_array[i][j]

# 将该城市一级指标数据交给遗传算法训练 ， 学习得到FCM权重array

result_of_GA = genetic_algorithm.genetic_algorithm(second_level_indicator_data)

# 将遗传算法得到的该城市FCM权重array转为权重matrix
fcm_weights = np.zeros((104,104))
for m in range(104):
    for n in range(104):
        if n == m:
            fcm_weights[m][n] = 0
        elif n < m:
            fcm_weights[m][n] = result_of_GA[m * 103 + n]
        elif n > m:
            fcm_weights[m][n] = result_of_GA[m * 103 + n - 1]


# 将fcm_weight输出到excel：
writer = pd.ExcelWriter("fcm_weights.xlsx")
write_fcm_weights = pd.DataFrame(fcm_weights)
write_fcm_weights.to_excel(writer,"fcm_weights",float_format="%.3f")
writer.save()

def Sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
# initial_vector =second_level_indicator_data.mean(axis=1)
initial_vector =second_level_indicator_data[:,-1]
fcm_iteration = np.zeros((30, 104))

for i in range(30):
    initial_vector = initial_vector + np.dot(initial_vector, fcm_weights)
    initial_vector = Sigmoid(initial_vector)
    fcm_iteration[i, :] = initial_vector

writer = pd.ExcelWriter("fcm_iteration_second_indicator.xlsx")
write_fcm_iteration_second_indicator = pd.DataFrame(fcm_iteration)
write_fcm_iteration_second_indicator.to_excel(writer,"fcm_iteration_second_indicator",float_format="%.3f")
writer.save()


iteration_result = fcm_iteration[-1,:]

print("second_level_indicator_data计算权重")
entropy_weights = excelextract.entropy_weight_method(second_level_indicator_data)
score = np.zeros(8)
for i in range(8):
    score[i]=np.dot(iteration_result[i*13:i*13+13],entropy_weights[i*13:i*13+13])


score_dict ={}

for i in range(len(city1)):
    score_dict[city1[i]]=score[i]
sorted_score =sorted(score_dict.items(),key=lambda item:item[1],reverse=True)
for i in range(len(sorted_score)):
    print(sorted_score[i])

print("均值计算权重")
entropy_weights = excelextract.entropy_weight_method(rod.three_D_array_mean)
score = np.zeros(8)
for i in range(8):
    score[i]=np.dot(iteration_result[i*13:i*13+13],entropy_weights)


score_dict ={}

for i in range(len(city1)):
    score_dict[city1[i]]=score[i]
sorted_score =sorted(score_dict.items(),key=lambda item:item[1],reverse=True)
for i in range(len(sorted_score)):
    print(sorted_score[i])