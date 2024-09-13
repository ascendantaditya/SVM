import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from svm import main, predict_for_year
import nbformat
from nbconvert import HTMLExporter
import logging
import base64
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model, scaler, and dataset
model, scaler, dataset = main()

# Streamlit app
st.set_page_config(page_title="Energy Consumption Prediction", layout="wide")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About", "Jupyter Notebook"],
        icons=["house", "info-circle", "book"],
        menu_icon="cast",
        default_index=0,
    )

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

# Home Page
if selected == "Home":
    st.title("Energy Consumption Prediction")
    st.markdown("""
        Welcome to the Energy Consumption Prediction app. 
        Enter a year to predict the energy consumption for that year.
    """)
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
elif selected == "About":
    st.title("About")
    st.markdown("""
        ## Energy Consumption Prediction App
        This app predicts the energy consumption for a given year using a Support Vector Machine (SVM) model.
        
        **GitHub**: [SVM REPO](https://github.com/AkshanshAnant/SVM-ENERGY-PREDICT)
        
        **Authors**: [Akshansh Anant](https://github.com/AkshanshAnant) & [Aditya Tomar](https://github.com/ascendantaditya)
    """)

# Jupyter Notebook Page
elif selected == "Jupyter Notebook":
    st.title("Jupyter Notebook")
    st.markdown("### SVM Model Notebook")
    display_notebook("svm.ipynb")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117; /* Streamlit's dark blue background */
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #0E1117; /* Streamlit's dark blue background */
        color: white;
    }
    .stTextInput, .stNumberInput, .stButton, .stMarkdown, .stDataFrame, .stExpander, .stHeader, .stSubheader {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and display sidebar image
img_path = "sidebar.jpg"
img_base64 = img_to_base64(img_path)
if img_base64:
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}">',
        unsafe_allow_html=True,
    )