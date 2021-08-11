import pandas as pd
import numpy as np


def get_recommendations(df, name, cosine_sim, indices):
    # Get the index of the recipe
    idx = indices[name]
    print(name)
    # Get the pairwsie similarity scores of all recipes with the current recipe
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort recipes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar recipes
    sim_scores = sim_scores[1:4]

    # Get recipes indices
    recipe_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['name'].iloc[recipe_indices]


def search_recipes(df, food, features_list, ingredients="ingredients_string", tags="tags_string"):
    food_1 = f"{food}"
    food_2 = ""
    food_3 = ""
    
    # 1. Get recipes by ingredient
    reg_string= f"(?=.*{food_1})(?=.*{food_2})(?=.*{food_3})"
    mask = np.column_stack([df[ingredients].str.contains(reg_string, na=False)])
    ingredients_search = df[mask]
    
    print("Ingredients: ",len(ingredients_search))
    
    if len(ingredients_search) >= 1:
        
        # 2 Filter new recipes by tags
        query_1= f"{features_list[0]}"
        query_2= f"{features_list[1]}"
        query_3= ""

        # Search for ingredients
        reg_string= f"(?=.*{query_1})(?=.*{query_2})(?=.*{query_3})"
        mask = np.column_stack([ingredients_search[tags].str.contains(reg_string, na=False)])
        new_recipes = ingredients_search[mask]
        print("Tags: ",len(new_recipes))
    else:
        new_recipes = ingredients_search
        
    return new_recipes, ingredients_search