'''
login是“登陆信息表”，包含：用户ID，登录日期，登录具体时间
stuStudyTime是“当日人均学习时长表”，包含：用户ID，用户开始学习时间，用户结束学习时间
stuStudySchedule 是“学习行为活跃情况”，包含：用户ID，班次ID，学习类型，学习日期
class是"班级表"，包含：班次ID，班次名称
studentRegister是"学生注册表"，包含：用户ID,创建时间

学习行为类型;
page；仅点击网页
video:观看学习视频
Topic:完成课题
course;完成课程内容
assignment:完成课程作业
quiz:完成测试

分析思路：
从平台使用情况，学生习惯分析，课程质量评估三个方面
平台使用情况：每日登录次数，用户活跃度
习惯分析:分时段学习人数、学习行为次数、平均学习时长
课程质量评估:用户活跃度，学习行为次数，平均学习时长
'''

'''
#将MySQL文件转为csv文件
# 连接到 MySQL 数据库
connection = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456',
    database='stuanalysis',
    charset='utf8'  # 设置字符集为utf8
)
# 要导出的表名列表
table_names = ['zt_class', 'zt_login', 'zt_stu_study_schedule','zt_stu_study_time_beihang','zt_student_num']
# 导出的 CSV 文件存储路径
output_directory = 'data1'
# 逐个表导出为 CSV 文件
for table_name in table_names:
    # 从数据库中读取数据到 DataFrame
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con=connection)
    # 设置 CSV 文件路径
    csv_file_path = f"{output_directory}{table_name}.csv"
    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_file_path, index=False)
#修改每个表的文件名
# 关闭数据库连接
connection.close()'''

'''
#数据清洗
# 原CSV文件夹路径
csv_folder_path = 'data1'

# 获取文件夹中所有CSV文件的文件名
csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]

# 循环处理每个CSV文件
for file in csv_files:
    file_path = os.path.join(csv_folder_path, file)
    # 读取CSV文件为DataFrame
    df = pd.read_csv(file_path)

    # 检查缺失值并处理
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f'发现 {missing_values} 个缺失值')


        # 用平均值填充缺失值
        df.fillna(df.mean(), inplace=True)

    # 检查重复值并处理
    #duplicate_rows = df.duplicated().sum()
    #if duplicate_rows > 0:
     #   print(f'发现 {duplicate_rows} 个重复值')
        # 删除重复行
      #  df.drop_duplicates(inplace=True)

    # 处理异常值，例如移除超出3倍标准差的异常值
    std = df.std()
    mean = df.mean()
    outlier_threshold = 3
    df = df[~((df - mean).abs() > outlier_threshold * std).any(axis=1)]

    # 保存清洗后的结果为新的CSV文件
    cleaned_file_path = os.path.join('data2', 'cleaned_' + file)
    df.to_csv(cleaned_file_path, index=False)
    

    print(f'文件 {file} 数据清洗完成')'''
