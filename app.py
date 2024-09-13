import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from svm import main, predict_for_year
import nbformat
from nbconvert import HTMLExporter
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model, scaler, and dataset
model, scaler, dataset = main()

# Streamlit app configuration
st.set_page_config(page_title="Energy Consumption Prediction", layout="wide")

st.title("Energy Consumption Prediction")
st.markdown("""
    <style>
    h1 {
        color: white;
    }
    </style>
    Welcome to the Energy Consumption Prediction app. 
    Enter a year to predict the energy consumption for that year.
""", unsafe_allow_html=True)

# Sidebar for additional sections
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About", "Jupyter Notebook"])

# Function to convert Jupyter notebook to HTML
def convert_notebook_to_html(notebook_path):
    with open(notebook_path, 'r') as f:
        notebook_content = f.read()
    notebook = nbformat.reads(notebook_content, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    return body

# Function to display Jupyter notebook
def display_notebook(notebook_path):
    html_content = convert_notebook_to_html(notebook_path)
    st.components.v1.html(html_content, height=800, scrolling=True)

# Function to convert image to base64
def img_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

# Prediction Page
if page == "Prediction":
    st.header("Enter Year for Prediction")
    year = st.number_input("Year", min_value=2023, max_value=2073, value=2023)

    if st.button("Predict"):
        try:
            predicted_values, dates = predict_for_year(year, model, scaler, dataset)
            True_MegaWatt = dataset[dataset.index.year == year]["AEP_MW"].to_list()[:len(predicted_values)]
            Predicted_MegaWatt = [x[0] for x in predicted_values]
            Machine_Df = pd.DataFrame(data={
                "Date": dates,
                "TrueMegaWatt": True_MegaWatt,
                "PredictedMeagWatt": Predicted_MegaWatt
            })
            st.subheader(f"Predicted Energy Consumption for the Year {year}")
            st.dataframe(Machine_Df.style.highlight_max(axis=0))
            fig, ax = plt.subplots()
            ax.plot(Machine_Df["Date"], Machine_Df["TrueMegaWatt"], color="green", label="True MegaWatt")
            ax.plot(Machine_Df["Date"], Machine_Df["PredictedMeagWatt"], color="red", label="Predicted MegaWatt")
            plt.xlabel('Dates')
            plt.ylabel("Power in MW")
            plt.title(f"Energy Consumption Prediction for Year {year}")
            plt.legend()
            st.pyplot(fig)
        except ValueError as e:
            st.error(f"Error: {e}")

# About Page
elif page == "About":
    st.header("About")
    st.markdown("""
        ## Energy Consumption Prediction App
        This app predicts the energy consumption for a given year using a Support Vector Machine (SVM) model.
        
        **GitHub**: [Your GitHub Repository](https://github.com/your-repo)
        
        **Author**: Your Name
    """)

# Jupyter Notebook Page
elif page == "Jupyter Notebook":
    st.header("Jupyter Notebook")
    st.markdown("### SVM Model Notebook")
    display_notebook("svm.ipynb")

# Custom CSS for dark mode styling
st.markdown("""
    <style>
    .main {
        background-color: black;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: black;
        color: white;
    }
    .stTextInput, .stNumberInput, .stButton, .stMarkdown, .stDataFrame, .stExpander, .stHeader, .stSubheader {
        color: white !important;
    }
    .stDataFrame {
        background-color: black;
    }
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 45px;
    }
    .css-1inwz65 {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and display sidebar image
img_path = "imgs/sidebar_streamly_avatar.png"
img_base64 = img_to_base64(img_path)
if img_base64:
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
