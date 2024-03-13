import os
import streamlit as st

from hanshu import 每日登录人数, 学生平均登录时间时间与对应人数表, 不同学习时长段的学习人数统计, 每日活跃学生人数, \
    每日学习类型次数, 学生登录人次表, jueceshu, 相关数据表


def Layouts_plotly(Double_coordinates=None):
    st.sidebar.write('基于机器学习在线平台学习行为分析')
    add_selectbox = st.sidebar.radio(
        "学习行为分析数据可视化",
        ('相关数据表', "学生登录人次表",  "每日登录人数", "学生平均登录时间时间与对应人数表",
         "不同学习时长段的学习人数统计表", "每日活跃学生人数表", "每日学习类型次数表","成绩预测决策树")
    )
    if add_selectbox == "学生登录人次表":
        学生登录人次表()

    elif add_selectbox == "每日登录人数":
        每日登录人数()
    elif add_selectbox == "学生平均登录时间时间与对应人数表":
        学生平均登录时间时间与对应人数表()
    elif add_selectbox == "不同学习时长段的学习人数统计表":
        不同学习时长段的学习人数统计()
    elif add_selectbox == "每日活跃学生人数表":
        每日活跃学生人数()

    elif add_selectbox == "每日学习类型次数表":
        每日学习类型次数()
    elif add_selectbox == '相关数据表':
        相关数据表()
    elif add_selectbox == "成绩预测决策树":
        jueceshu()

    # 补充表单
    # st.sidebar.button('基本数据表', on_click=Double_coordinates)


def main():
    Layouts_plotly()


if __name__ == "__main__":
    main()
