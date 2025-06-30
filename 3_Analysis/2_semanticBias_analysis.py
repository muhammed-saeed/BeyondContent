import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data(file_path):
    """Load the CSV data"""
    df = pd.read_csv(file_path, delimiter=',')
    return df

def calculate_bias_metrics(df):
    """Calculate male representation percentage for each condition"""
    df = df.copy()
    
    for condition in ['English', 'Chinese', 'Native']:
        male_col = f'Overall_{condition}_Male'
        female_col = f'Overall_{condition}_Female'
        total_col = f'Total_{condition}'
        male_rep_col = f'{condition}_Male_Rep'
        
        df[total_col] = df[male_col] + df[female_col]
        df[male_rep_col] = (df[male_col] / df[total_col]).fillna(0)
    
    return df

def perform_statistical_tests(df, category):
    """Perform statistical tests for a specific category"""
    category_data = df[df['Bias_Type'] == category].copy()
    
    if len(category_data) == 0:
        return None
    
    results = []
    
    # Define model order
    model_order = ['flux', 'ideogram', 'dalle3']
    
    # Group by language and model, ensuring proper order
    for language in sorted(category_data['Language'].unique()):
        for model in model_order:
            # Get data for this language-model combination
            group = category_data[(category_data['Language'] == language) & 
                                (category_data['T2I_Model'] == model)]
            
            if len(group) == 0:
                continue
            
            # Separate masculine and feminine words
            masc_data = group[group['Grammar_Gender'] == 'male']
            fem_data = group[group['Grammar_Gender'] == 'female']
            
            row_data = {
                'Language': language,
                'Model': model,
            }
            
            # For each condition, calculate means
            for condition in ['English', 'Chinese', 'Native']:
                male_rep_col = f'{condition}_Male_Rep'
                
                # Masculine words
                if len(masc_data) > 0:
                    masc_mean = masc_data[male_rep_col].mean()
                    masc_fem_mean = 1 - masc_mean  # Female representation
                    row_data[f'{condition}_M_Male'] = masc_mean
                    row_data[f'{condition}_M_Female'] = masc_fem_mean
                    row_data[f'{condition}_M_n'] = len(masc_data)
                else:
                    row_data[f'{condition}_M_Male'] = np.nan
                    row_data[f'{condition}_M_Female'] = np.nan
                    row_data[f'{condition}_M_n'] = 0
                
                # Feminine words
                if len(fem_data) > 0:
                    fem_mean = fem_data[male_rep_col].mean()
                    fem_fem_mean = 1 - fem_mean  # Female representation
                    row_data[f'{condition}_F_Male'] = fem_mean
                    row_data[f'{condition}_F_Female'] = fem_fem_mean
                    row_data[f'{condition}_F_n'] = len(fem_data)
                else:
                    row_data[f'{condition}_F_Male'] = np.nan
                    row_data[f'{condition}_F_Female'] = np.nan
                    row_data[f'{condition}_F_n'] = 0
            
            # Perform statistical tests (Native vs English for both genders)
            if len(masc_data) > 1:
                try:
                    t_stat_m, p_val_m = stats.ttest_rel(masc_data['Native_Male_Rep'], masc_data['English_Male_Rep'])
                    row_data['Masc_pval'] = p_val_m
                except:
                    row_data['Masc_pval'] = np.nan
            else:
                row_data['Masc_pval'] = np.nan
                
            if len(fem_data) > 1:
                try:
                    t_stat_f, p_val_f = stats.ttest_rel(fem_data['Native_Male_Rep'], fem_data['English_Male_Rep'])
                    row_data['Fem_pval'] = p_val_f
                except:
                    row_data['Fem_pval'] = np.nan
            else:
                row_data['Fem_pval'] = np.nan
            
            results.append(row_data)
    
    return pd.DataFrame(results) if results else None

def format_significance(p_val):
    """Format p-value as significance stars"""
    if pd.isna(p_val):
        return ""
    elif p_val < 0.001:
        return "$^{***}$"
    elif p_val < 0.01:
        return "$^{**}$"
    elif p_val < 0.05:
        return "$^{*}$"
    else:
        return ""

def generate_category_table(df, category, save_path):
    """Generate LaTeX table for a specific category"""
    
    results_df = perform_statistical_tests(df, category)
    
    if results_df is None or len(results_df) == 0:
        print(f"No data for category: {category}")
        return
    
    # Map model names for display
    model_map = {'flux': 'Flux', 'ideogram': 'Ideogram', 'dalle3': 'DALL-E 3'}
    lang_map = {'german': 'DE', 'russian': 'RU', 'italian': 'IT', 'french': 'FR', 'spanish': 'ES'}
    
    latex_content = []
    
    # Table header
    latex_content.append(f"% Table for {category.upper().replace('_', ' ')} category")
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("    \\centering")
    latex_content.append("    \\scriptsize")
    latex_content.append("    \\setlength{\\tabcolsep}{2pt}")
    latex_content.append("    \\begin{tabular}{@{}clccccccc@{}}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\multirow{2}{*}{\\textbf{Lang}} & \\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{1}{c}{\\textbf{Grammar}} & \\multicolumn{2}{c}{\\textbf{English}} & \\multicolumn{2}{c}{\\textbf{Chinese}} & \\multicolumn{2}{c}{\\textbf{Native}} \\\\")
    latex_content.append("        && \\multicolumn{1}{c}{\\textbf{Gender}} & \\multicolumn{2}{c}{\\textbf{Prompt}} & \\multicolumn{2}{c}{\\textbf{Prompt}} & \\multicolumn{2}{c}{\\textbf{Prompt}} \\\\")
    latex_content.append("        \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}")
    latex_content.append("        &&& M \\%  & F \\% & M \\%  & F \\% & M \\%  & F \\% \\\\")
    latex_content.append("        \\midrule")
    
    # Group by language
    for language in results_df['Language'].unique():
        lang_data = results_df[results_df['Language'] == language]
        lang_short = lang_map.get(language, language.upper())
        
        first_lang_row = True
        
        for _, row in lang_data.iterrows():
            model_name = model_map.get(row['Model'], row['Model'])
            
            # Add language label for first row of each language
            if first_lang_row:
                lang_label = f"\\multirow{{{len(lang_data)*2}}}{{*}}{{{lang_short}}}"
                first_lang_row = False
            else:
                lang_label = ""
            
            # Check if we have both masculine and feminine data
            has_masculine = not pd.isna(row['Native_M_Male']) and row['Native_M_n'] > 0
            has_feminine = not pd.isna(row['Native_F_Male']) and row['Native_F_n'] > 0
            
            if has_masculine:
                # Masculine row
                masc_sig = format_significance(row['Masc_pval'])
                
                latex_content.append(f"        {lang_label} & \\multirow{{2}}{{*}}{{{model_name}}} & M & "
                                   f"{row['English_M_Male']:.2f} & {row['English_M_Female']:.2f} & "
                                   f"{row['Chinese_M_Male']:.2f} & {row['Chinese_M_Female']:.2f} & "
                                   f"\\hlblue{{{row['Native_M_Male']:.2f}{masc_sig}}} & {row['Native_M_Female']:.2f} \\\\")
                lang_label = ""  # Clear for subsequent rows
            
            if has_feminine:
                # Feminine row
                fem_sig = format_significance(row['Fem_pval'])
                
                latex_content.append(f"        {lang_label} & & F & "
                                   f"{row['English_F_Male']:.2f} & {row['English_F_Female']:.2f} & "
                                   f"{row['Chinese_F_Male']:.2f} & {row['Chinese_F_Female']:.2f} & "
                                   f"{row['Native_F_Male']:.2f} & \\hlred{{{row['Native_F_Female']:.2f}{fem_sig}}} \\\\")
            
            # Add separator between models within the same language
            if row.name < len(lang_data) - 1:  # Not the last row for this language
                latex_content.append("        \\cmidrule(lr){2-9}")
        
        # Add separator between languages
        if language != results_df['Language'].unique()[-1]:  # Not the last language
            latex_content.append("        \\midrule")
    
    # Table footer
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    
    caption_text = (f"Gender representation for {category.replace('_', ' ').title()} category. "
                   f"\\hlblue{{Blue: masculine grammar}}; \\hlred{{red: feminine grammar}}. "
                   f"Significance: *p<.05, **p<.01, ***p<.001.")
    
    latex_content.append(f"    \\caption{{\\footnotesize {caption_text}}}")
    latex_content.append(f"    \\label{{tab:{category}_category}}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # Save to file
    filename = f"table_{category}_category.tex"
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✅ Generated table for {category}: {filename}")
    
    # Print sample sizes for reference
    print(f"   Sample sizes in {category}:")
    # Count unique words properly
    unique_masc = len([word for word in results_df['Language'].unique() 
                      if len(df[(df['Bias_Type'] == category) & 
                               (df['Grammar_Gender'] == 'male') & 
                               (df['Language'] == word)]['Native_Word'].unique()) > 0])
    unique_fem = len([word for word in results_df['Language'].unique() 
                     if len(df[(df['Bias_Type'] == category) & 
                              (df['Grammar_Gender'] == 'female') & 
                              (df['Language'] == word)]['Native_Word'].unique()) > 0])
    
    # Actually, let's get the total unique words properly
    category_data = df[df['Bias_Type'] == category]
    total_unique_masc = len([word for word in category_data['Native_Word'].unique() 
                            if category_data[category_data['Native_Word'] == word].iloc[0]['Grammar_Gender'] == 'male'])
    total_unique_fem = len([word for word in category_data['Native_Word'].unique() 
                           if category_data[category_data['Native_Word'] == word].iloc[0]['Grammar_Gender'] == 'female'])
    
    print(f"   - Unique masculine words: {total_unique_masc}")
    print(f"   - Unique feminine words: {total_unique_fem}")
    print(f"   - Total observations (words × models × languages): {len(category_data)}")
    
    return '\n'.join(latex_content)

def generate_all_category_tables(file_path, save_path):
    """Generate separate tables for each category"""
    
    # Load and process data
    df = load_data(file_path)
    df = calculate_bias_metrics(df)
    
    print("🔍 GENERATING CATEGORY-SPECIFIC TABLES")
    print("="*50)
    
    # First, let's understand the actual data structure
    print("\n📊 DATASET ANALYSIS:")
    print(f"Total rows: {len(df)}")
    print(f"Unique words: {df['Native_Word'].nunique()}")
    print(f"Languages: {df['Language'].nunique()} ({', '.join(df['Language'].unique())})")
    print(f"Models: {df['T2I_Model'].nunique()} ({', '.join(df['T2I_Model'].unique())})")
    
    # Check unique words per category (this is the correct way)
    print("\n📈 UNIQUE WORDS PER CATEGORY:")
    for category in df['Bias_Type'].unique():
        # Get unique words for this category
        category_unique_words = df[df['Bias_Type'] == category]['Native_Word'].unique()
        
        # Count masculine and feminine words
        masc_words = []
        fem_words = []
        
        for word in category_unique_words:
            word_data = df[df['Native_Word'] == word].iloc[0]  # Get first occurrence
            if word_data['Grammar_Gender'] == 'male':
                masc_words.append(word)
            elif word_data['Grammar_Gender'] == 'female':
                fem_words.append(word)
        
        print(f"  {category}: {len(category_unique_words)} total ({len(masc_words)} masc, {len(fem_words)} fem)")
    
    categories = df['Bias_Type'].unique()
    os.makedirs(save_path, exist_ok=True)
    
    all_tables = []
    
    for category in categories:
        print(f"\n📊 Processing {category.upper().replace('_', ' ')} category...")
        
        # Get unique words for this category
        category_unique_words = df[df['Bias_Type'] == category]['Native_Word'].unique()
        
        # Count masculine and feminine unique words
        masc_unique_words = []
        fem_unique_words = []
        
        for word in category_unique_words:
            word_data = df[df['Native_Word'] == word].iloc[0]  # Get first occurrence to check gender
            if word_data['Grammar_Gender'] == 'male':
                masc_unique_words.append(word)
            elif word_data['Grammar_Gender'] == 'female':
                fem_unique_words.append(word)
        
        print(f"   Unique words - Masculine: {len(masc_unique_words)}, Feminine: {len(fem_unique_words)}")
        
        if len(masc_unique_words) == 0 and len(fem_unique_words) == 0:
            print(f"   ⚠️  Skipping {category} - no words present")
            continue
        
        # Generate table even if only one gender (like relationship_descriptors)
        print(f"   📝 Generating table for {category}...")
        print(f"       (Has masculine: {len(masc_unique_words) > 0}, Has feminine: {len(fem_unique_words) > 0})")
        
        table_latex = generate_category_table(df, category, save_path)
        all_tables.append(table_latex)
    
    # Generate combined file with all tables
    combined_file = os.path.join(save_path, "all_category_tables.tex")
    with open(combined_file, 'w') as f:
        f.write("% All category-specific tables\n")
        f.write("% Generated for grammatical gender bias analysis\n\n")
        f.write('\n\n'.join(all_tables))
    
    print(f"\n📝 All tables saved to: {save_path}")
    print(f"📄 Combined file: all_category_tables.tex")
    
    return all_tables

# Main execution
if __name__ == "__main__":
    file_path = "./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/matched_rows.csv"
    save_path = "./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/codesResponseToREviewr/semanticData2"
    
    try:
        tables = generate_all_category_tables(file_path, save_path)
        
        print("\n🎯 SUMMARY:")
        print("="*30)
        print("✅ Category-specific tables generated")
        print("✅ Each table shows within-category effects")
        print("✅ Addresses reviewer concern about confounding")
        print("✅ Ready for LaTeX compilation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()