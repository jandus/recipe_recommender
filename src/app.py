import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import pickle

import modules.prediction as predict_model
from modules.recommender import get_recommendations, search_recipes

st.set_option('deprecation.showfileUploaderEncoding', False)

def save_uploaded_file(uploadedfile):
  with open(os.path.join("../data/tmp",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  #return #st.success("Saved file :{} in tempDir".format(uploadedfile.name))


st.title('PICK AND COOK')

DATE_COLUMN = 'date/time'


# Load model

# create file uploader
img_data = st.file_uploader(label='Upload an image of a Fruit or Vegetable', type=['jpg', 'jpeg'])



if img_data is not None:
    #Display image
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img,width = 150)
    save_uploaded_file(img_data)
    img_path = f"../data/tmp/{img_data.name}"

    predictions, probabilities = predict_model.predict_image(img_data.name)

    st.markdown("## Fruit and Vegetable probability")
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if float(eachProbability) > 60:
            st.write("   ", eachPrediction , " : " , eachProbability)


    # Load cosine_similarity matrix
    filename = '../data/models/content-based_filtering/cosine_sim_recipes.pickle'
    cosine_sim = pickle.load(open(filename, 'rb'))
    recommend_recipes = pd.read_csv('../data/recipe_recommender/recommend_recipes.csv')

    indices = pd.read_csv('../data/recipe_recommender/indices.csv', header = None, index_col=0, squeeze=True)

    item = f"{predictions[0]}".lower()
    features_list = ["", ""]
    new_recipes, only_ingredients = search_recipes(recommend_recipes, item, features_list)


    if len(new_recipes) >= 1:
        print("With all characteristics")
        recipe = new_recipes.sample(n=1, random_state=1)
        # recommend 3
        recipe_rec = get_recommendations(recommend_recipes, recipe["name"].iloc[0], cosine_sim, indices)
    elif len(only_ingredients) >= 1:
        print("entro a only ingredients")
        recipe = only_ingredients.sample(n=1, random_state=1) #Select random recipe
        #recommend 3 more and specify that it is only based on ingredients
        recipe_rec = get_recommendations(recommend_recipes, recipe["name"].iloc[0], cosine_sim, indices)
    else:
        st.write("No results found")

    st.markdown(f'## Recipe with {item.title()}')
    st.write("**Name:** ", recipe["name"].iloc[0].title())
    st.write("**Ingredients:** ", '\r', recipe["ingredients"].iloc[0].title())
    st.write("**Preparation Steps:** ", '\r', recipe["steps"].iloc[0].title())
    st.write("**Tags:** ", '\r', recipe["tags"].iloc[0])
    # st.write(recipe[['name', 'tags', 'ingredients', 'steps']].iloc[:3])
    # st.markdown(f'**Similar recipes recommendation:**')
    # st.write(recipe_rec[['name', 'tags', 'ingredients', 'steps']].iloc[:3])
    st.markdown(f'## Similar recipes recommendation')
    st.write("**Name:** ", recipe_rec.iloc[0].title())
    st.write("**Ingredients:** ", '\r', recommend_recipes[recommend_recipes['name']==recipe_rec.iloc[0]]['ingredients'].iloc[0].title())
    st.write("**Preparation Steps:** ", '\r', recommend_recipes[recommend_recipes['name']==recipe_rec.iloc[0]]['steps'].iloc[0].title())
    st.write("**Tags:** ", '\r', recommend_recipes[recommend_recipes['name']==recipe_rec.iloc[0]]["tags"].iloc[0])
