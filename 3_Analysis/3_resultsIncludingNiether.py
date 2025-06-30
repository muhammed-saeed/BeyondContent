import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/matched_rows.csv')

# Clean column names
df.columns = df.columns.str.strip()

def calculate_table_excluding_neither(df):
    """
    Calculate Table 1 EXCLUDING neither category from denominator
    Format: Male% = Male / (Male + Female)
            Female% = Female / (Male + Female)
    This is the original approach that excludes "neither" responses
    """
    
    languages = ['german', 'russian', 'italian', 'french', 'spanish']
    models = ['flux', 'ideogram', 'dalle3']
    lang_codes = {'german': 'DE', 'russian': 'RU', 'italian': 'IT', 'french': 'FR', 'spanish': 'ES'}
    
    results = []
    
    for lang in languages:
        for model in models:
            # Filter data for current language and model
            subset = df[(df['Language'] == lang) & (df['T2I_Model'] == model)]
            
            for grammar_gender in ['male', 'female']:
                gender_subset = subset[subset['Grammar_Gender'] == grammar_gender]
                
                row_data = {
                    'Language': lang_codes[lang],
                    'Model': model.capitalize(),
                    'Grammar_Gender': 'M' if grammar_gender == 'male' else 'F'
                }
                
                # Calculate for each context (English, Chinese, Native/Gendered)
                # NOTE: Excluding neither from denominator
                contexts = [
                    ('English', 'Overall_English_Male', 'Overall_English_Female'),
                    ('Chinese', 'Overall_Chinese_Male', 'Overall_Chinese_Female'),
                    ('Native', 'Overall_Native_Male', 'Overall_Native_Female')
                ]
                
                for context_name, male_col, female_col in contexts:
                    # Sum across all words for this condition
                    total_male = gender_subset[male_col].sum()
                    total_female = gender_subset[female_col].sum()
                    
                    # Calculate total EXCLUDING neither
                    total_binary = total_male + total_female
                    
                    if total_binary > 0:
                        male_pct = total_male / total_binary
                        female_pct = total_female / total_binary
                    else:
                        male_pct = female_pct = 0
                    
                    row_data[f'{context_name}_M'] = male_pct
                    row_data[f'{context_name}_F'] = female_pct
                
                results.append(row_data)
    
    return pd.DataFrame(results)

def calculate_significance_excluding_neither(df, lang, model, grammar_gender, context1, context2):
    """Calculate statistical significance between two contexts excluding neither"""
    subset = df[(df['Language'] == lang) & (df['T2I_Model'] == model) & (df['Grammar_Gender'] == grammar_gender)]
    
    # Get data for both contexts (excluding neither)
    male_col1 = f'Overall_{context1}_Male'
    female_col1 = f'Overall_{context1}_Female' 
    
    male_col2 = f'Overall_{context2}_Male'
    female_col2 = f'Overall_{context2}_Female'
    
    # Create arrays for chi-square test
    context1_data = []
    context2_data = []
    
    for _, row in subset.iterrows():
        # Context 1 proportions (excluding neither)
        male1 = row[male_col1]
        female1 = row[female_col1] 
        total1 = male1 + female1
        
        # Context 2 proportions (excluding neither)
        male2 = row[male_col2]
        female2 = row[female_col2]
        total2 = male2 + female2
        
        if total1 > 0 and total2 > 0:
            context1_data.extend([0] * male1 + [1] * female1)
            context2_data.extend([0] * male2 + [1] * female2)
    
    if len(context1_data) > 0 and len(context2_data) > 0:
        try:
            # Chi-square test
            contingency = pd.crosstab(
                context1_data + context2_data,
                [1] * len(context1_data) + [2] * len(context2_data)
            )
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            return p_value
        except:
            return 1.0
    return 1.0

def format_table_excluding_neither_latex(results_df, df_original):
    """Format Table 1 as LaTeX excluding neither category"""
    
    latex_lines = []
    latex_lines.append("\\begin{table}[h!]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\scriptsize % Smaller font size")
    latex_lines.append("    \\setlength{\\tabcolsep}{2pt} % Tighter column spacing")
    latex_lines.append("    \\begin{tabular}{@{}clccccccc@{}}")
    latex_lines.append("        \\toprule")
    latex_lines.append("        \\multirow{2}{*}{\\textbf{Lang}} & \\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{1}{c}{\\textbf{Grammar}} & \\multicolumn{2}{c}{\\textbf{English}} & \\multicolumn{2}{c}{\\textbf{Chinese}} & \\multicolumn{2}{c}{\\textbf{Gendered}} \\\\")
    latex_lines.append("        && \\multicolumn{1}{c}{\\textbf{Gender}} & \\multicolumn{2}{c}{\\textbf{Prompt}} & \\multicolumn{2}{c}{\\textbf{Prompt}} & \\multicolumn{2}{c}{\\textbf{Prompt}} \\\\")
    latex_lines.append("        \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}")
    latex_lines.append("        &&& M \\%  & F \\% & M \\%  & F \\% & M \\%  & F \\% \\\\")
    latex_lines.append("        \\midrule")
    
    lang_mapping = {'DE': 'german', 'RU': 'russian', 'IT': 'italian', 'FR': 'french', 'ES': 'spanish'}
    model_mapping = {'Flux': 'flux', 'Ideogram': 'ideogram', 'Dalle3': 'dalle3'}
    
    current_lang = None
    
    for _, row in results_df.iterrows():
        lang = row['Language']
        model = row['Model']
        gender = row['Grammar_Gender']
        
        # Add language group header
        if current_lang != lang:
            if current_lang is not None:
                latex_lines.append("        \\midrule")
            current_lang = lang
        
        # Calculate significance for Native vs English comparison
        lang_full = lang_mapping[lang]
        model_full = model_mapping[model]
        gender_full = 'male' if gender == 'M' else 'female'
        
        p_value = calculate_significance_excluding_neither(df_original, lang_full, model_full, gender_full, 'English', 'Native')
        
        # Determine significance stars
        if p_value < 0.001:
            sig_stars = "***"
        elif p_value < 0.01:
            sig_stars = "**"
        elif p_value < 0.05:
            sig_stars = "*"
        else:
            sig_stars = ""
        
        # Format percentages
        eng_m = f"{row['English_M']:.2f}"
        eng_f = f"{row['English_F']:.2f}"
        chi_m = f"{row['Chinese_M']:.2f}"
        chi_f = f"{row['Chinese_F']:.2f}"
        nat_m = f"{row['Native_M']:.2f}"
        nat_f = f"{row['Native_F']:.2f}"
        
        # Add highlighting based on expected bias direction
        if gender == 'M':  # Male grammar should increase male representation
            if row['Native_M'] > row['English_M']:
                nat_m = f"\\hlblue{{{nat_m}{'$^{' + sig_stars + '}$' if sig_stars else ''}}}"
            else:
                nat_f = f"\\hlred{{{nat_f}{'$^{' + sig_stars + '}$' if sig_stars else ''}}}"
        else:  # Female grammar should increase female representation  
            if row['Native_F'] > row['English_F']:
                nat_f = f"\\hlred{{{nat_f}{'$^{' + sig_stars + '}$' if sig_stars else ''}}}"
            else:
                nat_m = f"\\hlblue{{{nat_m}{'$^{' + sig_stars + '}$' if sig_stars else ''}}}"
        
        # Check if English shows higher female representation than gendered
        if row['English_F'] > row['Native_F'] and gender == 'F':
            eng_f = f"\\textcolor{{brown}}{{{eng_f}}}"
        
        # Build the row
        if gender == 'M':
            row_str = f"        \\multirow{{2}}{{*}}{{{lang}}} & \\multirow{{2}}{{*}}{{{model}}} & {gender} & {eng_m} & {eng_f} & {chi_m} & {chi_f} & {nat_m} & {nat_f} \\\\"
        else:
            row_str = f"        & & {gender} & {eng_m} & {eng_f} & {chi_m} & {chi_f} & {nat_m} & {nat_f} \\\\"
            
        latex_lines.append(row_str)
        
        # Add model separator if this is the second row of a model group
        if gender == 'F':
            latex_lines.append("        \\cmidrule(lr){2-9}")
    
    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("    \\caption{\\footnotesize Gender representation percentages EXCLUDING 'neither' category (M=Male, F=Female) across languages (DE=German, RU=Russian, IT=Italian, FR=French, ES=Spanish) and models. \\hlblue{Blue: masculine grammar increases male representation}; \\hlred{red: feminine grammar increases female representation}; \\textcolor{brown}{brown}: English prompts showing higher female representation than gendered prompts. Significance levels: *p<.05, **p<.01, ***p<.001. The second highest statistical significance between English and Chinese baselines is highlighted for comparison with gendered prompts.}")
    latex_lines.append("    \\vspace{-2em}")
    latex_lines.append("    \\label{tab:grammatical_gender_effects_excluding_neither}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

def compare_with_without_neither(df):
    """
    Compare results with and without neither category to show the impact
    """
    print("=== COMPARISON: WITH vs WITHOUT NEITHER CATEGORY ===")
    print("This analysis shows how excluding 'neither' responses affects the results\n")
    
    # Calculate both versions
    results_with_neither = calculate_table_1_with_neither(df)
    results_without_neither = calculate_table_excluding_neither(df)
    
    print("Sample comparison for first few rows:")
    print("Format: Language-Model-Gender | With Neither | Without Neither | Difference")
    print("-" * 80)
    
    for i in range(min(6, len(results_with_neither))):
        row_with = results_with_neither.iloc[i]
        row_without = results_without_neither.iloc[i]
        
        lang = row_with['Language']
        model = row_with['Model']
        gender = row_with['Grammar_Gender']
        
        # Compare Native context male percentages
        with_nat_m = row_with['Native_M']
        without_nat_m = row_without['Native_M']
        diff_m = without_nat_m - with_nat_m
        
        # Compare Native context female percentages
        with_nat_f = row_with['Native_F']
        without_nat_f = row_without['Native_F']
        diff_f = without_nat_f - with_nat_f
        
        print(f"{lang}-{model}-{gender} Native Male  | {with_nat_m:.3f} | {without_nat_m:.3f} | {diff_m:+.3f}")
        print(f"{lang}-{model}-{gender} Native Female| {with_nat_f:.3f} | {without_nat_f:.3f} | {diff_f:+.3f}")
        print()
    
    print("Key Insight: Excluding 'neither' responses typically increases both male and female percentages")
    print("since the denominator becomes smaller (Male + Female only vs Male + Female + Neither)")
    print("=" * 80 + "\n")

def calculate_table_1_with_neither(df):
    """
    Calculate Table 1 with neither category included in denominator
    (Reusing function from previous code for comparison)
    """
    
    languages = ['german', 'russian', 'italian', 'french', 'spanish']
    models = ['flux', 'ideogram', 'dalle3']
    lang_codes = {'german': 'DE', 'russian': 'RU', 'italian': 'IT', 'french': 'FR', 'spanish': 'ES'}
    
    results = []
    
    for lang in languages:
        for model in models:
            subset = df[(df['Language'] == lang) & (df['T2I_Model'] == model)]
            
            for grammar_gender in ['male', 'female']:
                gender_subset = subset[subset['Grammar_Gender'] == grammar_gender]
                
                row_data = {
                    'Language': lang_codes[lang],
                    'Model': model.capitalize(),
                    'Grammar_Gender': 'M' if grammar_gender == 'male' else 'F'
                }
                
                contexts = [
                    ('English', 'Overall_English_Male', 'Overall_English_Female', 'Overall_English_Neither'),
                    ('Chinese', 'Overall_Chinese_Male', 'Overall_Chinese_Female', 'Overall_Chinese_Neither'),
                    ('Native', 'Overall_Native_Male', 'Overall_Native_Female', 'Overall_Native_Neither')
                ]
                
                for context_name, male_col, female_col, neither_col in contexts:
                    total_male = gender_subset[male_col].sum()
                    total_female = gender_subset[female_col].sum()
                    total_neither = gender_subset[neither_col].sum()
                    
                    total_all = total_male + total_female + total_neither
                    
                    if total_all > 0:
                        male_pct = total_male / total_all
                        female_pct = total_female / total_all
                    else:
                        male_pct = female_pct = 0
                    
                    row_data[f'{context_name}_M'] = male_pct
                    row_data[f'{context_name}_F'] = female_pct
                
                results.append(row_data)
    
    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    print("=== GENERATING TABLE EXCLUDING 'NEITHER' CATEGORY ===")
    print("This table uses the original methodology: Male% = Male/(Male+Female)\n")
    
    # Show comparison first
    compare_with_without_neither(df)
    
    # Calculate Table 1 excluding neither
    table_results = calculate_table_excluding_neither(df)
    table_latex = format_table_excluding_neither_latex(table_results, df)
    
    # Print results
    print("=== TABLE: Gender Representation EXCLUDING Neither Category ===")
    print(table_latex)
    print("\n")
    
    # Save to file
    with open('table_excluding_neither.tex', 'w') as f:
        f.write(table_latex)
    
    # Save CSV for inspection
    table_results.to_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/codesResponseToREviewr/oldTable/table_excluding_neither_data.csv', index=False)
    
    print("Files saved:")
    print("- table_excluding_neither.tex")
    print("- table_excluding_neither_data.csv")
    print("\nThis table can be used to show consistency with your original methodology")
    print("while the 'with neither' table demonstrates methodological transparency.")