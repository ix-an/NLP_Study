# ragas_simple_viewer.py
import streamlit as st
import pandas as pd
from pathlib import Path

# 页面基础配置（简洁版）
st.set_page_config(
    page_title="RAGAS Demo Results",
    page_icon="📊",
    layout="wide"  # 宽屏展示表格更清晰
)


# -------------------------- 读取CSV文件 --------------------------
@st.cache_data  # 缓存数据，提升加载速度
def load_ragas_data():
    """读取当前目录下的ragas_result.csv"""
    file_path = Path("ragas_result.csv")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"读取文件出错：{str(e)}")
        return None


# -------------------------- 核心展示逻辑 --------------------------
# 主标题
st.title("RAGAS_demo_results")

# 读取数据
df = load_ragas_data()

# 展示数据
if df is not None:
    # 简单的数据集信息提示
    st.info(f"📝 共加载 {len(df)} 条样本，{len(df.columns)} 个字段")

    # 展示完整CSV数据（自适应宽度）
    st.dataframe(df, use_container_width=True)
else:
    # 文件不存在时的友好提示
    st.warning("未找到 ragas_result.csv 文件！")
    st.info("请确保该文件与本脚本放在同一目录下")

# 极简页脚
st.markdown("---")
st.caption("RAGAS 结果查看器 | 仅展示CSV原始数据")