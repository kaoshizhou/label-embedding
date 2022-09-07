import os
import re
import csv
import docx

ipc_label_path = '/mnt/sdb/nlp/ipc/out_5y/train'
ipc_label_file = open(os.path.join(ipc_label_path, 'ipc_labels.txt'), 'r')
labels = ipc_label_file.read()
label_list = labels.split('\n')[:-1]
pos = [0, 84, 249, 334, 372, 402, 501, 585, 635]

ipc_def_path = '/home/user/ipc/IPC/'
file_list = []  # 生成文档列表
for file_name in os.listdir(ipc_def_path):
    fn = ipc_def_path + file_name
    file_list.append(fn)

# 定义写入函数，包含去除多余文本信息的规则
def write(inputlabel, inputtext):
    if not inputtext:  # 如果定义为空则不写入
        return
    mytext = inputtext.strip()  # 去掉开头和结尾的空白字符

    if mytext[-1] == '〕':  # 去掉定义最后的〔5〕标注
        left = mytext.rfind('〔')
        if left != -1:  # 如果文本中存在〔
            mytext = mytext[:left]

    if mytext[-1] == ']':  # 去掉定义最后的[2006.01]及[5]之类的标注
        left = mytext.rfind('[')
        if left != -1:  # 如果文本中存在[
            mytext = mytext[:left]  # 删除标注

    # 将“不包含”“不包括”“未列入”“其他类目”这些词删掉
    remove_list = ['本小类', '其他单独小类中的', '其他类目中的', '其他小类中', '其他组中', '其他相关子类目', '其他相关子类', '小类中的', '其他类目中',
                   '其他类目的', '其他类目', '其他位置的', '其他类', '不包含的', '不包括的', '不包含在', '不包括在', '未包含在', '所包含的', '未包括', '未列入']
    for word in remove_list:
        mytext = mytext.replace(word, "")

    mytext = mytext.replace('/', "")  # 删除文本中的/
    mode = r'\W\w+入[A-Z]\d\d\w*'  # 设置正则表达式，用于识别"（xxx入xxx"，"，xxx入xxx"
    mytext_new = re.sub(mode, "", mytext)  # 删除符合正则表达式的句子（只删除了左边的标点）
    if mytext_new != mytext:  # 定义中包含正则表达式
        mytext = mytext_new
        mytext = re.sub(r'\W+', "，", mytext)  # 删除多余的标点，并将所有标点改为逗号

    mytext = re.sub(r'在?[A-Z]\d\d\w*[至,或][A-Z]\d\d\w*[组,小类]内?的?', "", mytext)  # 删除“在A01P100至A01P1300组”这种句子
    mytext = re.sub(r'与[A-Z]\d\d\w*组有关的', "", mytext)
    mytext = re.sub(r'[A-Z]\d\d\w*优先', "", mytext)
    mytext = re.sub(r'[A-Z]\d\d[A-Z]?(\d{3,7})?', "", mytext)  # 删除定义中所有ipc标号
    mytext = re.sub(r'\W+', "，", mytext)  # 删除多余的标点，并将所有标点改为逗号
    if mytext and mytext[-1] == '，':
        mytext = mytext[:-1]  # 去掉最后的逗号

    with open(os.path.join(ipc_label_path, "ipc_def_7w.csv"), "a+", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([inputlabel, mytext])

for i in range(len(file_list)):
    fn = file_list[i]
    doc = docx.Document(fn)
    table = doc.tables[0]
    nrow = len(table.rows)
    labels = label_list[pos[i]:pos[i + 1]]  # 取出每个字母开头的所有IPC label
    temp_row = 0  # temp_row表示当前遍历到文档的第几行
    for label in labels:
        count = 0
        if label == 'F24J':  # 该ipc不在2022国际专利分类标准中
            write(label, '不包含在其他类目中的热量产生和利用（所用材料入C09K 5/00；发动机或其他由热产生机械动力的机械装置见有关类，例如利用自然界热量的入F03G）')
            print('{} has 1 definition'.format(label))
            continue

        while temp_row < nrow:
            row = table.rows[temp_row]
            temp = row.cells[0].text
            if temp == label:
                count += 1
                cell = row.cells[2]  # 读取表格第三列，即三级定义的所有内容
                text = cell.text
                text_to_write = text.split('\n')[0]  # 读取定义的第一段
                write(temp, text_to_write)
                temp_row += 1
                next_row = table.rows[temp_row]
                next_temp = next_row.cells[0].text
                while len(next_temp) > 4 or next_temp == "":  # 如果当前行是该类下属大组或小组的定义，或当前行的第一个元素为空
                    text_to_write = next_row.cells[2].text  # 读取定义
                    write(temp, text_to_write)
                    count += 1
                    temp_row += 1
                    next_row = table.rows[temp_row]
                    next_temp = next_row.cells[0].text
                print('{} has {} definitions'.format(temp, count))
                break
            temp_row += 1
