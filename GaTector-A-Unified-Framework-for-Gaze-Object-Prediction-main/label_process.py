import os
import json

# # folder_path = "./txt"  # 替换为你的文件夹路径
# # output_file = './txt/gt.txt'  # 替换为你想要的输出文件名
# folder_path = "./test_data/test_img/label"  # 替换为你的文件夹路径
# output_file = './test_data/test_img/label/test_gt.txt'  # 替换为你想要的输出文件名
#
# txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]
#
# with open(output_file, 'w') as output:
#     # 遍历每个txt文件
#     for txt_file in txt_files:
#         file_path = os.path.join(folder_path, txt_file)
#         img_name=txt_file.replace(".txt",".jpg")
#         with open(file_path, 'r') as file:
#             # 读取每一行的内容并写入输出文件，加上空格
#             lines = file.readlines()
#
#             for line in lines:
#                 new_line = img_name + " " + line
#                 output.write(new_line.strip() + '\n')




directory = './data/img_all/label'

# # 创建空列表或字典作为最终结果容器
# result = []  # 如果需要保存为列表
result = [] # 如果需要保存为字典
output_data={}
# #遍历目录下的所有文件
# for filename in os.listdir(directory):
#     if filename.endswith('.json'):
#         file_path = os.path.join(directory, filename)
#
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#             result.append(data)
#             # 根据需求进行相应操作，这里只是简单地将数据添加到结果容器中
# output_data["data"]=result
# output_file = './data/img_all/label/train_gt.json'
# with open(output_file, 'w') as file:
#     json.dump(output_data, file)


output_file = './data/img_all/label/train_gt.json'
with open(output_file, 'r') as file:
    data = json.load(file)
    print(1)




