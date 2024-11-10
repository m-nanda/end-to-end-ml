import streamlit as st
import os, json, joblib, requests
from inference_pipeline import *
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def visual_projection(df_train:pd.DataFrame, x: float, y: float) -> None:
  """
  Visualize trained data in feature engineering projections with user input data
  point to support prediction result's confidence.

  Args:
      df_train (pd.DataFrame): trained data
      x (float): sepal_width-coordinate of the user input data.
      y (float): petal_width-coordinate of the user input data.

  Returns:
      None
  """

  # Existing data
  fig = px.scatter(
      df_train,
      x=df_train.columns[1],
      y=df_train.columns[2],
      color=df_train.columns[0],
      title='Visual Projection'
  )
  
  # Prediction
  new_point = go.Scatter(
      x=[x],  
      y=[y],  
      mode='markers',
      marker=dict(symbol='x', size=12, color='black'),
      name='User Input Data'  
  )
  fig.add_trace(new_point)
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def main_page():
  """
  Main page for prediction
  """
  # load credential
  # load_dotenv()
  ML_API_V1 = os.getenv('ML_API_V1')
  ML_API_V2 = os.getenv('ML_API_V2')

  # Load existing data from pipeline
  df_train = joblib.load(f'{SAVED_OBJECTS_PATH}/data_feature_engineering.bin') 

  st.set_page_config(page_title='Iris-Classification', layout='wide')

  st.title('Predicting Iris Flower using Machine Learning')
  st.markdown("""<div style="text-align: justify;"> The Iris flower dataset was first introduced in 1936 by the British statistician Ronald Fisher in his paper 
  <a href=https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x>"The Use of Multiple Measurements in Taxonomic Problems."</a> Fisher collected the data as part of his work on linear discriminant analysis, a statistical method for predicting the class of an object based on its measurements. Nowadays, this dataset is commonly used in machine learning, particularly for classification tasks. The dataset contains measurements of the sepal length, sepal width, petal length, and petal width of three species of iris flowers: Iris setosa, Iris versicolor, and Iris virginica. The goal of the classification task is to predict the species of iris flower based on these measurements. The dataset is small, with only 150 instances, making it a good dataset for testing and comparing the performance of different classification algorithms. Additionally, the dataset is well-balanced, with 50 instances of each species, ensuring that each class is equally represented in the data. This web app predicts Iris Flower as a case of machine learning deployment in production.</div>""", unsafe_allow_html=True)
  st.image('https://content.codecademy.com/programs/machine-learning/k-means/iris.svg', 'Iris Flower (image source: kaggle.com/code/necibecan/iris-dataset-eda-n)')

  with st.sidebar:
    use_feature_engineering = st.radio(
      "Use Feature Engineering",
      (True, False)
    )
    model = st.radio(
      "Choose Model",
      ("SVM", "LogisticRegression")
    )

  if use_feature_engineering:
    model = model+"_FE"
    
  tab1, tab2 = st.tabs(['Custom', 'Existing (by index)'])

  with tab1:
    st.write('Fill Characteristics')

    # get input data from user
    sepal_length = st.number_input('Sepal Length (cm):', step=0.01, min_value=0.0, format='%.f')
    sepal_width = st.number_input('Sepal Width (cm):', step=0.01, min_value=0.0, format='%.f') 
    petal_length = st.number_input('Petal Length (cm):', step=0.01, min_value=0.0, format='%.f')
    petal_width = st.number_input('Petal Width (cm):', step=0.01, min_value=0.0, format='%.f') 
    
    submitted = st.button('Predict')
    if submitted:
      
      # call API
      headers = {"Content-Type": "application/json"}
      body = {
        "data":{
          "sepal_length": sepal_length,
          "sepal_width": sepal_width,
          "petal_length": petal_length,
          "petal_width": petal_width
        },
        "model": model
      }
      results = requests.post(f"{ML_API_V2}", headers=headers, json=body)
      
      # show result from API
      results = json.loads(results.text)
      output_to_user = f"Based on the input provided, it is {results['prediction_proba']:.2f}% characteristic of the {results['prediction_str']} species"
      st.success(output_to_user)
      visual_projection(df_train,
                        sepal_length*sepal_width,
                        petal_length*petal_width)

      # clear result
      reset = st.button('Clear Results')	
      if reset:      
        submitted=False
        del results

  with tab2:
    st.write('Choose index')
    idx = st.selectbox('Index data:', (range(150)))

    try:
      data_tab2 = fetch_data()
      input_data_tab2 = data_tab2.loc[[int(idx)]]
    except Exception as e:
      st.exception(e)

    submitted_tab2 = st.button(label='Predict ', key='Predict_by_idx')
    if submitted_tab2:
      st.write('Input Data:')
      st.dataframe(input_data_tab2)

      # call API
      headers = {"Content-Type": "application/json"}
      body = {
        "index":idx,
        "model":model
      }
      results = requests.post(f"{ML_API_V1}", headers=headers, json=body)

      # show result from API
      results = json.loads(results.text)

      output_to_user = f"Based on the input provided, it is {results['prediction_proba']:.2f}% characteristic of the {results['prediction_str']} species"
      st.success(output_to_user)

      visual_projection(df_train,
                        input_data_tab2[["sepal_length","sepal_width"]].prod(axis=1).values[0],
                        input_data_tab2[["petal_length","petal_width"]].prod(axis=1).values[0])

      # clear result
      reset_tab2 = st.button('Clear Results')	
      if reset_tab2:      
        submitted_tab2=False
        del results

  footer = st.container()
  with footer:
    st.write('<hr>', unsafe_allow_html=True)
    st.write('<h6 style="text-align:center";><i>Author: <a href="https://www.linkedin.com/in/muhammadnanda">Muhammad Nanda</a></i></h6>', 
             unsafe_allow_html=True)

if __name__ == "__main__":
  main_page()