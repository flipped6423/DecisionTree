import streamlit as st
import bisect
import os

import mplcursors as mplcursors
import pandas as pd
import pymysql
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.dates import DayLocator, DateFormatter


def jueceshu():
    import numpy as np
    import pandas as pd1
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    import graphviz
    import pydotplus
    import matplotlib.pyplot as plt1

    # 加载数据
    data = pd1.read_csv('data2/student_data.csv')
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前5条')
    st.table(data.head(5))

    # 数据清洗，采用用平均值填充空缺值
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)
    data = np.array(data.values)
    feature = data[:, 0:3]
    label = data[:, 3]
    for i in range(np.size(label)):
        if label[i] < 60:
            label[i] = 3
        elif 80 > label[i] > 59:
            label[i] = 2
        else:
            label[i] = 1

    # 数据集划分，70%训练数据，30%测试数据
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3)

    # 选取最合适的深度
    max_depths = []
    for max_depth in range(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth + 1)
        clf.fit(feature_train, label_train)  # 拟合
        score = clf.score(feature_test, label_test)
        max_depths.append(score)
    best_depth = max_depths.index(max(max_depths)) + 1
    plt1.figure(figsize=(20, 8), dpi=80)
    plt1.plot(range(1, 11), max_depths)
    plt1.xlabel('max depth')
    plt1.ylabel('evaluate score')
    plt1.show()
    st.pyplot(plt1)

    # 选取最合适的最小叶子树
    min_samples = []
    for min_sample in range(30):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, min_samples_leaf=min_sample + 5)
        clf.fit(feature_train, label_train)  # 拟合
        score = clf.score(feature_test, label_test)
        min_samples.append(score)
    best_min_samples_leaf = min_samples.index(max(min_samples)) + 5
    plt1.figure(figsize=(20, 8), dpi=80)
    plt1.plot(range(4, 34), min_samples)
    plt1.xlabel('min samples leaf')
    plt1.ylabel('evaluate score')
    plt1.show()
    st.pyplot(plt1)

    # 根据最合适的参数构建模型
    mytree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_depth,
                                         min_samples_leaf=best_min_samples_leaf)
    mytree.fit(feature_train, label_train)

    # 可视化
    dot_data = tree.export_graphviz(mytree, out_file=None, \
                                    feature_names=["Attendance", "Preview", "Job"], \
                                    class_names=["excellent", "good", "poor"], \
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("dtree10.pdf")

    # 计算预测正确率
    rate = np.sum(mytree.predict(feature_test) == label_test) / mytree.predict(feature_test).size
    # print('训练集数量：', label_train.size)
    # print('测试集数量：', label_test.size)
    # print('正确率：', rate)

    # 可视化决策树
    plt1.figure(figsize=(20, 20))
    tree.plot_tree(mytree, filled=True, feature_names=["Attendance", "Preview", "Job"],
                   class_names=["excellent", "good", "poor"])
    plt1.show()
    st.pyplot(plt1)
    st.write('训练集数量：', label_train.size)
    st.write('测试集数量：', label_test.size)
    st.write('正确率：', rate)


def 相关数据表():
    files = os.listdir("data2")

    # 创建一个 Streamlit 侧边栏组件，用于显示文件列表
    selected_file = st.sidebar.selectbox("Select a file", files)

    # 读取所选文件的数据
    df = pd.read_csv(os.path.join("data2", selected_file))

    # 显示数据
    st.write(df)


def 学生登录人次表():
    # 登录次数和人数柱状图：登录次数在5次及其以下的居多，达到了将近2000个，而5-10次的就减为了一半
    zt_login = pd.read_csv("data2/cleaned_login.csv")
    # 在侧边栏显示完整数据表
    # st.sidebar.subheader('基本数据表')
    # st.sidebar.write(zt_login)
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前十条')
    st.table(zt_login.head(10))

    user_info = zt_login['user_id'].value_counts().reset_index()
    user_info.columns = ['user', 'times']

    breakpoints = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    count_list = [0 for _ in range(len(breakpoints) + 1)]

    for item in user_info['times']:
        level_index = bisect.bisect_right(breakpoints, item)
        count_list[level_index] += 1

    breakpoints.insert(0, 0)

    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.bar([x + 5 for x in breakpoints], count_list)
    plt.title("学生登录人次表")
    plt.xlabel("登录次数")
    plt.ylabel("登录人数")
    plt.show()
    st.pyplot(plt)

    '''
   # 绘制饼图
   plt.pie(count_list[1:], labels=[f'{breakpoints[i]}-{breakpoints[i + 1]}' for i in range(len(breakpoints) - 1)],
           autopct='%1.1f%%')
   plt.title("学生登录人次分布")
   plt.show()
   st.pyplot(plt)
   '''


def 每日登录人数():
    # 按照登录日期统计每日登录人数：登录人数在2月15日达到了顶峰，2月5日次之。而2月的平均登录人数在739人次。
    # 读取CSV文件
    df = pd.read_csv('data2/cleaned_login.csv')
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前5条')
    st.table(df.head(5))
    # 按照登录日期统计每日登录人数
    daily_login_count = df['login_date'].value_counts().sort_index()
    # 将日期转换为字符串
    daily_login_count.index = daily_login_count.index.astype(str)
    # 绘制折线图
    # daily_login_count['login_date'] = daily_login_count['login_date'].astype(str)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=daily_login_count.index, y=daily_login_count.values, marker='o')
    plt.title('每日登录人数')
    plt.xlabel('日期')
    plt.ylabel('人数')
    plt.xticks(rotation=90)
    # 标注数据点
    for x, y in zip(daily_login_count.index, daily_login_count.values):
        plt.text(x, y, f'{y}', ha='right', va='bottom')
    plt.show()
    st.pyplot(plt)


def 学生平均登录时间时间与对应人数表():
    # 读取原始数据
    data = pd.read_csv("data2/cleaned_stuStudyTime.csv")
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前5条')
    st.table(data.head(5))

    # 创建空字典存储学习时间和学习次数
    study_time_info = {}
    study_count_info = {}

    # 遍历原始数据，计算每个用户的学习时间和次数
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        if user_id not in study_time_info:
            last_visit_time = pd.to_datetime(data["last_visit_time"][i])
            visit_time = pd.to_datetime(data["visit_time"][i])
            delta_minutes = (last_visit_time - visit_time).total_seconds() / 60  # 将秒数转换为分钟
            study_time_info[user_id] = delta_minutes
            study_count_info[user_id] = 1
        else:
            last_visit_time = pd.to_datetime(data["last_visit_time"][i])
            visit_time = pd.to_datetime(data["visit_time"][i])
            delta_minutes = (last_visit_time - visit_time).total_seconds() / 60  # 将秒数转换为分钟
            study_time_info[user_id] += delta_minutes
            study_count_info[user_id] += 1

    # 计算每个用户的平均学习时间
    user_id_list = []
    count_list = []
    total_time_list = []
    avg_time_list = []

    for key, value in study_time_info.items():
        user_id_list.append(key)
        total_time_list.append(value)
        count = study_count_info[key]
        count_list.append(count)
        avg_time_list.append(value / count)
        '''
           # 创建DataFrame并保存为CSV文件
    result = pd.DataFrame({
        "user_id": user_id_list,
        "total_time": total_time_list,
        "count": count_list,
        "avg_time": avg_time_list
    })
    result.to_csv("用户平均登录时间表.csv", index=False)
        '''
    # 对用户平均登录时间进行可视化
    avg_time_count = {}

    for item in avg_time_list:
        if int(item) not in avg_time_count:
            avg_time_count[int(item)] = 1
        else:
            avg_time_count[int(item)] += 1

    avg_time_list = list(avg_time_count.keys())
    avg_time_list.sort()

    count_list = [avg_time_count[x] for x in avg_time_list]

    break_points = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960]
    group_count = [0 for _ in break_points]

    for item in avg_time_list:
        level_index = bisect.bisect_right(break_points, item)
        group_count[level_index] += 1

    # 绘制可视化图表
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.title("学生平均登录时间时间与对应人数")
    plt.xlabel("平均时长(分钟)")
    plt.ylabel("人数")
    plt.plot(break_points, group_count)
    plt.show()
    st.pyplot(plt)


def 不同学习时长段的学习人数统计():
    # 1.分时段学习人数

    # 读取 CSV 文件
    df = pd.read_csv('data2/cleaned_stuStudyTime.csv')
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前5条')
    st.table(df.head(5))

    # 将开始学习时间和最后学习时间转换为 datetime 类型
    df['开始学习时间'] = pd.to_datetime(df['visit_time'])
    df['最后学习时间'] = pd.to_datetime(df['last_visit_time'])

    # 计算每个用户的学习总时长
    df['学习时长'] = df['最后学习时间'] - df['开始学习时间']
    print(df['学习时长'].count())

    # 按用户ID分组并计算总学习时长
    total_study_time = df.groupby('user_id')['学习时长'].sum().reset_index()

    # 按照学习总时长降序排列
    sorted_total_study_time = total_study_time.sort_values(by='学习时长', ascending=False)

    print(sorted_total_study_time)
    '''
    #显示结果
    1098
          user_id            学习时长
    141  17682951 8 days 23:22:06
    94   15258984 5 days 14:21:44
    109  15259322 5 days 11:47:03
    33   13049829 4 days 20:58:36
    140  17639401 4 days 17:31:22
    ..        ...             ...
    2     8084284 0 days 00:28:49
    115  15932449 0 days 00:19:59
    179  19939998 0 days 00:14:20
    12   11995893 0 days 00:06:27
    214  20388108 0 days 00:00:45

    [239 rows x 2 columns]
    '''

    # 计算分时段的学习人数
    df['学习时长'] = (df['最后学习时间'] - df['开始学习时间']).dt.total_seconds() / 3600  # 学习时长（小时）

    # 根据学习时长进行分组统计
    bins = [0, 1, 2, 4, 8, 24, 168]  # 分时段边界（小时）
    labels = ['<1', '1-2', '2-4', '4-8', '8-24', '>24']  # 分时段标签
    df['学习时长段'] = pd.cut(df['学习时长'], bins=bins, labels=labels)

    # 统计每个分时段的学习人数
    study_time_counts = df['学习时长段'].value_counts().sort_index()
    # 可视化为饼图
    plt.figure(figsize=(10, 6))
    plt.pie(study_time_counts, labels=study_time_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("viridis", len(study_time_counts)))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('不同学习时长段的学习人数比例')
    plt.show()
    st.pyplot(plt)
    '''  # 可视化为面积图
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=study_time_counts.index, y=study_time_counts.values, palette="viridis", marker='o')
    plt.fill_between(study_time_counts.index, study_time_counts.values, color="skyblue", alpha=0.4)
    plt.title('不同学习时长段的学习人数统计')
    plt.xlabel('学习时长段（小时）')
    plt.ylabel('学习人数')
    plt.show()
    st.pyplot(plt)'''

    '''  # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x=study_time_counts.index, y=study_time_counts.values, palette="viridis")
    plt.title('不同学习时长段的学习人数统计')
    plt.xlabel('学习时长段（小时）')
    plt.ylabel('学习人数')
    plt.show()
    st.pyplot(plt)'''


def 每日活跃学生人数():
    # 课程质量评估:
    # 用户活跃度：通过查询zt_stu_study_schedule表来完成每日活跃学生人数的统计分析，这里设定每日至少进行3次学习行为的用户为活跃用户

    # 读取CSV文件到DataFrame
    df = pd.read_csv('data2/cleaned_stuStudySchedule.csv')
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前5条')
    st.table(df.head(5))
    # 按学习日期和用户ID进行分组，并计算每日活跃学生人数
    active_students_per_day = df.groupby(['updated_date'])['user_id'].nunique().reset_index()
    active_students_per_day.columns = ['学习日期', '活跃人数']
    # 打印每日活跃学生人数
    print(active_students_per_day)
    # 可视化
    active_students_per_day['学习日期'] = active_students_per_day['学习日期'].astype(str)
    plt.plot(active_students_per_day['学习日期'], active_students_per_day['活跃人数'], marker='o')
    plt.title('每日活跃学生人数')
    plt.xlabel('日期')
    plt.ylabel('活跃人数')
    x_ticks = active_students_per_day['学习日期'][::5]  # 每隔两天显示一个日期
    plt.xticks(x_ticks, rotation=45)
    plt.grid(True)

    plt.show()
    st.pyplot(plt)


def 每日学习类型次数():
    # 读取CSV文件
    df = pd.read_csv('data2/cleaned_stuStudySchedule.csv')
    # 在主界面显示前十条数据
    st.markdown('### 数据表显示 - 前5条')
    st.table(df.head(5))

    # 按照学习日期和学习类型进行分组，并计算每日总学习类型次数
    study_counts = df.groupby(['updated_date', 'content_type']).size().reset_index(name='学习次数')

    # 使用透视表将数据重塑为适合绘图的格式
    pivot_table = study_counts.pivot(index='updated_date', columns='content_type', values='学习次数').fillna(0)

    # 绘制叠加柱状图
    ax = pivot_table.plot(kind='bar', stacked=True, figsize=(12, 6))
    # 设置x轴日期格式
    ax.xaxis.set_major_locator(DayLocator(interval=5))  # 设置日期间隔为5天
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # 设置日期显示格式为年-月-日
    plt.title('每日学习类型次数')
    plt.xlabel('日期')
    plt.ylabel('学习次数')
    plt.legend(title='学习类型')
    plt.show()
    st.pyplot(plt)
