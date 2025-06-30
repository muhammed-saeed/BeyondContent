import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/matched_rows.csv')

# Clean column names
df.columns = df.columns.str.strip()

def calculate_bias_scores_fixed(df):
    """
    Calculate bias scores excluding 'neither' responses:
    Native vs English and Native vs Chinese only
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
                
                # Calculate totals for all three contexts (excluding neither)
                contexts = {
                    'English': {
                        'male': gender_subset['Overall_English_Male'].sum(),
                        'female': gender_subset['Overall_English_Female'].sum()
                    },
                    'Chinese': {
                        'male': gender_subset['Overall_Chinese_Male'].sum(),
                        'female': gender_subset['Overall_Chinese_Female'].sum()
                    },
                    'Native': {
                        'male': gender_subset['Overall_Native_Male'].sum(),
                        'female': gender_subset['Overall_Native_Female'].sum()
                    }
                }
                
                # Calculate percentages for each context (excluding neither)
                percentages = {}
                for context_name, counts in contexts.items():
                    total = counts['male'] + counts['female']  # Only male + female
                    if total > 0:
                        percentages[context_name] = {
                            'male_pct': counts['male'] / total,
                            'female_pct': counts['female'] / total
                        }
                    else:
                        percentages[context_name] = {
                            'male_pct': 0, 'female_pct': 0
                        }
                
                # Calculate bias scores (Native - English, Native - Chinese only)
                if all(ctx in percentages for ctx in ['English', 'Chinese', 'Native']):
                    results.append({
                        'Language': lang_codes[lang],
                        'Model': model.capitalize(),
                        'Grammar_Gender': 'M' if grammar_gender == 'male' else 'F',
                        # Native - English
                        'Native_vs_English_Male': percentages['Native']['male_pct'] - percentages['English']['male_pct'],
                        'Native_vs_English_Female': percentages['Native']['female_pct'] - percentages['English']['female_pct'],
                        # Native - Chinese
                        'Native_vs_Chinese_Male': percentages['Native']['male_pct'] - percentages['Chinese']['male_pct'],
                        'Native_vs_Chinese_Female': percentages['Native']['female_pct'] - percentages['Chinese']['female_pct']
                    })
    
    return pd.DataFrame(results)

def format_fixed_bias_table_latex(bias_df):
    """Format simplified bias score table as LaTeX"""
    
    latex_lines = []
    latex_lines.append("\\begin{table}[h!]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\scriptsize")
    latex_lines.append("    \\setlength{\\tabcolsep}{3pt}")
    latex_lines.append("    \\begin{tabular}{@{}lcc|cc|cc@{}}")
    latex_lines.append("        \\toprule")
    latex_lines.append("        \\multirow{2}{*}{\\textbf{Lang}} & \\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{Gram}} &")
    latex_lines.append("        \\multicolumn{2}{c|}{\\textbf{Native - English}} & \\multicolumn{2}{c}{\\textbf{Native - Chinese}} \\\\")
    latex_lines.append("        && & \\textbf{M} & \\textbf{F} & \\textbf{M} & \\textbf{F} \\\\")
    latex_lines.append("        \\midrule")
    
    current_lang = None
    
    for _, row in bias_df.iterrows():
        lang = row['Language']
        model = row['Model']
        grammar = row['Grammar_Gender']
        
        # Add language separator
        if current_lang != lang:
            if current_lang is not None:
                latex_lines.append("        \\midrule")
            current_lang = lang
        
        # Format bias scores with proper coloring logic
        def format_score_with_logic(score, is_male_score, is_masculine_grammar):
            formatted = f"{score*100:+.1f}"  # Convert to percentage points
            
            # Logic for expected vs counterintuitive effects
            if is_masculine_grammar:
                # Masculine grammar: expect +Male, -Female
                if (is_male_score and score > 0.05) or (not is_male_score and score < -0.05):
                    return f"\\textcolor{{green}}{{\\textbf{{{formatted}}}}}"  # Expected effect
                elif (is_male_score and score < -0.05) or (not is_male_score and score > 0.05):
                    return f"\\textcolor{{red}}{{{formatted}}}"  # Counterintuitive effect
                else:
                    return formatted  # Neutral
            else:
                # Feminine grammar: expect -Male, +Female
                if (is_male_score and score < -0.05) or (not is_male_score and score > 0.05):
                    return f"\\textcolor{{green}}{{\\textbf{{{formatted}}}}}"  # Expected effect
                elif (is_male_score and score > 0.05) or (not is_male_score and score < -0.05):
                    return f"\\textcolor{{red}}{{{formatted}}}"  # Counterintuitive effect
                else:
                    return formatted  # Neutral
        
        is_masculine = (grammar == 'M')
        
        # Format scores
        eng_m = format_score_with_logic(row['Native_vs_English_Male'], True, is_masculine)
        eng_f = format_score_with_logic(row['Native_vs_English_Female'], False, is_masculine)
        chi_m = format_score_with_logic(row['Native_vs_Chinese_Male'], True, is_masculine)
        chi_f = format_score_with_logic(row['Native_vs_Chinese_Female'], False, is_masculine)
        
        # Build the row with proper spacing
        if grammar == 'M':
            row_str = f"        \\multirow{{2}}{{*}}{{{lang}}} & \\multirow{{2}}{{*}}{{{model}}} & {grammar} & {eng_m} & {eng_f} & {chi_m} & {chi_f} \\\\"
        else:
            row_str = f"        & & {grammar} & {eng_m} & {eng_f} & {chi_m} & {chi_f} \\\\"
            
        latex_lines.append(row_str)
        
        # Add model separator after F row
        if grammar == 'F':
            latex_lines.append("        \\cmidrule(lr){2-7}")
    
    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("    \\caption{\\footnotesize Bias scores (percentage point differences) excluding 'neither' responses. M=Male, F=Female. \\textcolor{green}{Green}: bias in expected direction; \\textcolor{red}{Red}: bias in opposite direction. Bold: strongest expected effects.}")
    latex_lines.append("    \\label{tab:comprehensive_bias_scores}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

# Main execution
if __name__ == "__main__":
    print("=== CALCULATING BIAS SCORES ===")
    
    # Calculate fixed bias scores
    bias_results = calculate_bias_scores_fixed(df)
    
    # Display first few rows for verification
    print("First 10 rows of bias scores:")
    print(bias_results.head(10))
    
    # Generate LaTeX table
    bias_latex = format_fixed_bias_table_latex(bias_results)
    
    print("\n=== LATEX TABLE ===")
    print(bias_latex)
    
    # Save to files
    with open('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/codesResponseToREviewr/oldTable/fixed_bias_scores.tex', 'w') as f:
        f.write(bias_latex)
    
    bias_results.to_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/codesResponseToREviewr/oldTable/fixed_bias_scores_data.csv', index=False)
    
    print("\n=== FILES SAVED ===")
    print("- fixed_bias_scores.tex")
    print("- fixed_bias_scores_data.csv")
    
    # Show summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("Native vs English - Male bias:")
    print(f"  Mean: {bias_results['Native_vs_English_Male'].mean():.3f}")
    print(f"  Std:  {bias_results['Native_vs_English_Male'].std():.3f}")
    print(f"  Range: [{bias_results['Native_vs_English_Male'].min():.3f}, {bias_results['Native_vs_English_Male'].max():.3f}]")
    
    print("Native vs English - Female bias:")
    print(f"  Mean: {bias_results['Native_vs_English_Female'].mean():.3f}")
    print(f"  Std:  {bias_results['Native_vs_English_Female'].std():.3f}")
    print(f"  Range: [{bias_results['Native_vs_English_Female'].min():.3f}, {bias_results['Native_vs_English_Female'].max():.3f}]")