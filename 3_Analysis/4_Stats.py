import pandas as pd
import numpy as np

# Load the data into a pandas DataFrame
df = pd.read_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/ModelsOutputForPaperEvalution/gender_classification_summary.csv')

# Define a function to calculate bias
def calculate_bias(male_count, female_count):
    total = male_count + female_count
    if total == 0:
        return np.nan, np.nan  # Return NaN if total is zero to avoid division by zero
    male_bias = male_count / total
    female_bias = female_count / total
    return male_bias, female_bias

# --- Calculate Bias by Bias Type, Grammar Gender, Language, and Model with Prompting Language ---
bias_data_type_gender_lang_model_prompt = []
bias_types = df['Bias_Type'].unique()
grammar_genders = df['Grammar_Gender'].unique()
languages = df['Language'].unique()
models = df['T2I_Model'].unique()
prompting_languages = ['English', 'Chinese', 'Native']

for bias_type in bias_types:
    df_bias = df[df['Bias_Type'] == bias_type]
    for gender in grammar_genders:
        df_gender = df_bias[df_bias['Grammar_Gender'] == gender]
        for lang in languages:
            df_lang = df_gender[df_gender['Language'] == lang]
            row_data = {'Bias Type': bias_type, 'Grammar Gender': gender, 'Language': lang}
            for model in models:
                df_model = df_lang[df_lang['T2I_Model'] == model]
                for prompt_lang in prompting_languages:
                    male_col = f'Overall_{prompt_lang}_Male'
                    female_col = f'Overall_{prompt_lang}_Female'

                    overall_male = df_model[male_col].sum()
                    overall_female = df_model[female_col].sum()
                    male_bias, female_bias = calculate_bias(overall_male, overall_female)

                    row_data[f'{model} ({prompt_lang}) Male Bias'] = male_bias
                    row_data[f'{model} ({prompt_lang}) Female Bias'] = female_bias
            bias_data_type_gender_lang_model_prompt.append(row_data)

output_df_type_gender_lang_model_prompt = pd.DataFrame(bias_data_type_gender_lang_model_prompt)
output_df_type_gender_lang_model_prompt.to_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reports/bias_by_type_gender_language_model_prompt.csv', index=False)
print("Saved bias by bias type, grammar gender, language, and model with prompting language to 'bias_by_type_gender_language_model_prompt.csv'")