from contextlib import contextmanager
from io import StringIO
from threading import current_thread
import streamlit as st
import pandas as pd
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import base64
from streamlit.runtime.scriptrunner import get_script_run_ctx  # 替代旧版线程API

# 设置页面配置，页面标题为“Python期末项目”
st.set_page_config(page_title="破晓二部项目三")


@contextmanager
def st_redirect(src, dst):
    '''
    将打印输出重定向到Streamlit用户界面。
    替代已弃用的REPORT_CONTEXT_ATTR_NAME的现代实现。
    '''
    # 创建一个空的占位符
    placeholder = st.empty()
    # 获取占位符的输出函数
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        # 保存原始的写入函数
        old_write = src.write

        def new_write(b):
            # 检查是否在Streamlit上下文中（替代旧版线程检查）
            ctx = get_script_run_ctx()
            if ctx is not None:
                # 将内容写入缓冲区
                buffer.write(b)
                # 在Streamlit界面输出缓冲区内容
                output_func(buffer.getvalue())
            else:
                # 如果不在Streamlit上下文，使用原始写入函数
                old_write(b)

        try:
            # 替换原始写入函数
            src.write = new_write
            yield
        finally:
            # 恢复原始写入函数
            src.write = old_write


@contextmanager
def st_stdout(dst):
    '''
        子实现，用于重定向标准输出，提高代码可读性。
    '''
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    '''
        子实现，用于在出错时重定向标准错误输出，提高代码可读性。
    '''
    with st_redirect(sys.stderr, dst):
        yield


def _all_subdirs_of(b='.'):
    '''
        返回指定路径下的所有子目录。
    '''
    result = []
    # 遍历指定路径下的所有文件和文件夹
    for d in os.listdir(b):
        # 拼接完整路径
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            # 如果是目录，则添加到结果列表中
            result.append(bd)
    return result


def _get_latest_folder():
    '''
        返回runs\detect目录下最新的文件夹。
    '''
    return max(_all_subdirs_of(os.path.join('runs', 'detect')), key=os.path.getmtime)


# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加模型权重文件路径参数，默认值为best.pt
parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='模型权重文件路径')
# parser.add_argument('--weights', nargs='+', type=str, default='yolov5n6.pt', help='模型权重文件路径')
# 添加检测源参数，默认值为0，表示使用摄像头
parser.add_argument('--source', type=str, default='0', help='检测源（文件/文件夹，0表示摄像头）')
# 添加推理图像大小参数，默认值为640像素
parser.add_argument('--img-size', type=int, default=640, help='推理图像大小（像素）')
# 添加目标置信度阈值参数，默认值为0.25
parser.add_argument('--conf-thres', type=float, default=0.25, help='目标置信度阈值')
# 添加非极大值抑制的IOU阈值参数，默认值为0.45
parser.add_argument('--iou-thres', type=float, default=0.45, help='非极大值抑制的IOU阈值')
# 添加CUDA设备参数，默认值为空，表示自动选择
parser.add_argument('--device', default='', help='CUDA设备，例如0或0,1,2,3或cpu')
# 添加是否显示结果的参数
parser.add_argument('--view-img', action='store_true', help='显示检测结果')
# 添加是否保存检测结果到文本文件的参数
parser.add_argument('--save-txt', action='store_true', help='将检测结果保存到文本文件')
# 添加是否在保存的文本标签中保存置信度的参数
parser.add_argument('--save-conf', action='store_true', help='在保存的文本标签中保存置信度')
# 添加是否保存裁剪后的预测框的参数
parser.add_argument('--save-crop', action='store_true', help='保存裁剪后的预测框')
# 添加是否不保存图像/视频的参数
parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
# 添加按类别过滤的参数
parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤：--class 0，或 --class 0 2 3')
# 添加是否使用类别无关的非极大值抑制的参数
parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的非极大值抑制')
# 添加是否使用增强推理的参数
parser.add_argument('--augment', action='store_true', help='增强推理')
# 添加是否更新所有模型的参数
parser.add_argument('--update', action='store_true', help='更新所有模型')
# 添加保存检测结果的项目路径参数，默认值为runs/detect
parser.add_argument('--project', default='runs/detect', help='保存检测结果的项目路径')
# 添加保存检测结果的文件夹名称参数，默认值为exp
parser.add_argument('--name', default='exp', help='保存检测结果的文件夹名称')
# 添加是否允许使用已存在的项目/名称的参数
parser.add_argument('--exist-ok', action='store_true', help='允许使用已存在的项目/名称，不递增编号')
# 添加边界框厚度参数，默认值为3像素
parser.add_argument('--line-thickness', default=3, type=int, help='边界框厚度（像素）')
# 添加是否隐藏标签的参数
parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
# 添加是否隐藏置信度的参数
parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
# 解析命令行参数
opt = parser.parse_args()

# 定义检测源选项字典
CHOICES = {0: "上传图片", 1: "上传视频", 2: "使用摄像头"}


def _save_uploadedfile(uploadedfile):
    '''
        将上传的视频保存到磁盘。
    '''
    with open(os.path.join("data", "videos", uploadedfile.name), "wb") as f:
        # 将上传的视频内容写入文件
        f.write(uploadedfile.getbuffer())


def _format_func(option):
    '''
        用于选择框的键/值格式化函数。
    '''
    return CHOICES[option]


# 在侧边栏创建一个选择框，让用户选择检测源
inferenceSource = str(st.sidebar.selectbox('选择检测源:', options=list(CHOICES.keys()), format_func=_format_func))

if inferenceSource == '0':
    # 如果选择上传图片，在侧边栏创建一个文件上传器，允许上传png、jpeg、jpg格式的图片
    uploaded_file = st.sidebar.file_uploader("上传图片", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        # 如果上传了图片，标记为有效
        is_valid = True
        with st.spinner(text='正在处理'):
            # 在侧边栏显示上传的图片
            st.sidebar.image(uploaded_file)
            # 打开上传的图片
            picture = Image.open(uploaded_file)
            # 将图片保存到指定路径
            picture = picture.save(f'data/images/{uploaded_file.name}')
            # 设置检测源为保存的图片路径
            opt.source = f'data/images/{uploaded_file.name}'
    else:
        # 如果没有上传图片，标记为无效
        is_valid = False

elif inferenceSource == '1':
    # 如果选择上传视频，在侧边栏创建一个文件上传器，允许上传mp4格式的视频
    uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
    if uploaded_file is not None:
        # 如果上传了视频，标记为有效
        is_valid = True
        with st.spinner(text='正在处理'):
            # 在侧边栏播放上传的视频
            st.sidebar.video(uploaded_file)
            # 保存上传的视频到磁盘
            _save_uploadedfile(uploaded_file)
            # 设置检测源为保存的视频路径
            opt.source = f'data/videos/{uploaded_file.name}'
    else:
        # 如果没有上传视频，标记为无效
        is_valid = False

elif inferenceSource == '2':
    # 如果选择使用摄像头，标记为有效，并设置检测源为0
    is_valid = True
    opt.source = '0'

# 在页面上显示标题
st.title('欢迎使用人像识别系统')
# 在页面上显示副标题
st.subheader('基于YOLOv5的人体摔倒识别')

# 创建一个空的按钮占位符
inferenceButton = st.empty()

if is_valid:
    # 如果检测源有效，显示启动检测按钮
    if inferenceButton.button('启动检测！'):
        # 将标准输出重定向到Streamlit界面
        with st_stdout("info"):
            # 调用detect函数进行检测
            detect(opt)
        if inferenceSource == '1':
            # 如果选择的是上传视频，提示由于许可限制，部署版本无法播放视频
            st.warning('由于许可限制，部署版本无法播放视频。 ')
            with st.spinner(text='正在准备视频'):
                # 遍历最新的检测结果文件夹中的所有视频文件
                for vid in os.listdir(_get_latest_folder()):
                    # 在页面上播放视频
                    st.video(f'{_get_latest_folder()}/{vid}')
                # 播放气球动画
                st.balloons()
        elif inferenceSource == '0':
            with st.spinner(text='正在准备图片'):
                # 遍历最新的检测结果文件夹中的所有图片文件
                for img in os.listdir(_get_latest_folder()):
                    # 在页面上显示图片
                    st.image(f'{_get_latest_folder()}/{img}')
                # 播放气球动画
                st.balloons()
