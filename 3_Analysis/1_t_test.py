import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the original data with multiple observations
original_df = pd.read_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/matched_rows.csv')

# Define a function to perform t-tests
def perform_t_test(data, lang, model, gender):
    # Filter data to get samples for this language, model, and grammar gender
    filtered_data = data[(data['Language'] == lang) & 
                         (data['T2I_Model'] == model) & 
                         (data['Grammar_Gender'] == gender)]
    
    if len(filtered_data) < 2:
        return {'t_stat_eng': np.nan, 'p_value_eng': np.nan, 
                't_stat_zh': np.nan, 'p_value_zh': np.nan, 'sample_size': 0}
    
    # Calculate proportions for each sample
    if gender == 'male':
        # For male grammar gender, compare male representation
        native_values = filtered_data['Overall_Native_Male'] / (filtered_data['Overall_Native_Male'] + filtered_data['Overall_Native_Female'])
        english_values = filtered_data['Overall_English_Male'] / (filtered_data['Overall_English_Male'] + filtered_data['Overall_English_Female'])
        chinese_values = filtered_data['Overall_Chinese_Male'] / (filtered_data['Overall_Chinese_Male'] + filtered_data['Overall_Chinese_Female'])
    else:
        # For female grammar gender, compare female representation
        native_values = filtered_data['Overall_Native_Female'] / (filtered_data['Overall_Native_Female'] + filtered_data['Overall_Native_Male'])
        english_values = filtered_data['Overall_English_Female'] / (filtered_data['Overall_English_Female'] + filtered_data['Overall_English_Male'])
        chinese_values = filtered_data['Overall_Chinese_Female'] / (filtered_data['Overall_Chinese_Female'] + filtered_data['Overall_Chinese_Male'])
    
    # Remove NaN values
    valid_indices = ~(np.isnan(native_values) | np.isnan(english_values) | np.isnan(chinese_values))
    native_values = native_values[valid_indices]
    english_values = english_values[valid_indices]
    chinese_values = chinese_values[valid_indices]
    
    if len(native_values) < 2:
        return {'t_stat_eng': np.nan, 'p_value_eng': np.nan, 
                't_stat_zh': np.nan, 'p_value_zh': np.nan, 'sample_size': len(native_values)}
    
    # Perform paired t-tests
    t_stat_eng, p_value_eng = stats.ttest_rel(native_values, english_values)
    t_stat_zh, p_value_zh = stats.ttest_rel(native_values, chinese_values)
    
    return {
        't_stat_eng': t_stat_eng, 
        'p_value_eng': p_value_eng, 
        't_stat_zh': t_stat_zh, 
        'p_value_zh': p_value_zh,
        'sample_size': len(native_values)
    }

# Load the summary data
summary_df = pd.read_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/bias_by_language_model_gender.csv')

# Print header
print("\n===== STATISTICAL ANALYSIS OF GRAMMATICAL GENDER EFFECTS =====\n")

# Loop through languages
for lang in summary_df['Language'].unique():
    print(f"\n===== LANGUAGE: {lang.upper()} =====")
    
    # For each language, filter by grammar gender
    for gender in ['male', 'female']:
        gender_name = "MASCULINE" if gender == 'male' else "FEMININE"
        print(f"\n--- GRAMMAR GENDER: {gender_name} ---")
        
        # Filter by grammar gender
        gender_df = summary_df[(summary_df['Language'] == lang) & 
                              (summary_df['Grammar Gender'] == gender)]
        
        # For each model
        for _, row in gender_df.iterrows():
            model = row['Model']
            
            # For masculine grammar, we compare male representation
            # For feminine grammar, we compare female representation
            if gender == 'male':
                native_value = row['Native Male Bias']
                english_value = row['English Male Bias']
                chinese_value = row['Chinese Male Bias']
                comparison_type = "Male Representation"
            else:
                # For feminine grammar, we want female representation
                native_value = row['Native Female Bias']
                english_value = row['English Female Bias']
                chinese_value = row['Chinese Female Bias']
                comparison_type = "Female Representation"
            
            # Calculate differences
            native_eng_diff = native_value - english_value
            native_zh_diff = native_value - chinese_value
            
            # Determine effect direction
            sign_eng = "+" if native_eng_diff > 0 else "-"
            sign_zh = "+" if native_zh_diff > 0 else "-"
            
            # Perform t-test
            t_test_result = perform_t_test(original_df, lang, model, gender)
            
            # Format p-values with asterisks for significance
            p_eng = t_test_result['p_value_eng']
            p_zh = t_test_result['p_value_zh']
            
            sig_eng = ""
            if not np.isnan(p_eng):
                if p_eng < 0.001:
                    sig_eng = "***"
                elif p_eng < 0.01:
                    sig_eng = "**"
                elif p_eng < 0.05:
                    sig_eng = "*"
                    
            sig_zh = ""
            if not np.isnan(p_zh):
                if p_zh < 0.001:
                    sig_zh = "***"
                elif p_zh < 0.01:
                    sig_zh = "**"
                elif p_zh < 0.05:
                    sig_zh = "*"
            
            print(f"MODEL: {model}")
            print(f"  {comparison_type}:")
            print(f"    Native: {native_value:.4f}, English: {english_value:.4f}, Chinese: {chinese_value:.4f}")
            print(f"    Effect (Native-English): {native_eng_diff:.4f} {sign_eng} (t={t_test_result['t_stat_eng']:.3f}, p={p_eng:.5f} {sig_eng})")
            print(f"    Effect (Native-Chinese): {native_zh_diff:.4f} {sign_zh} (t={t_test_result['t_stat_zh']:.3f}, p={p_zh:.5f} {sig_zh})")
            print(f"    Sample size: {t_test_result['sample_size']}")

# Aggregate t-tests by grammar gender
print("\n\n===== AGGREGATED STATISTICAL TESTS: MASCULINE GRAMMAR =====")
masc_df = summary_df[summary_df['Grammar Gender'] == 'male']

# Count significant effects
sig_effects_eng_m = 0
sig_effects_zh_m = 0
total_m = 0

for lang in masc_df['Language'].unique():
    for model in masc_df[masc_df['Language'] == lang]['Model'].unique():
        t_test_result = perform_t_test(original_df, lang, model, 'male')
        if not np.isnan(t_test_result['p_value_eng']):
            total_m += 1
            if t_test_result['p_value_eng'] < 0.05:
                sig_effects_eng_m += 1
            
            if t_test_result['p_value_zh'] < 0.05:
                sig_effects_zh_m += 1

percent_sig_eng_m = (sig_effects_eng_m / total_m) * 100 if total_m > 0 else 0
percent_sig_zh_m = (sig_effects_zh_m / total_m) * 100 if total_m > 0 else 0

# Calculate average values
avg_native_m = masc_df['Native Male Bias'].mean()
avg_english_m = masc_df['English Male Bias'].mean()
avg_chinese_m = masc_df['Chinese Male Bias'].mean()

print(f"Average Male Representation:")
print(f"  Native: {avg_native_m:.4f}, English: {avg_english_m:.4f}, Chinese: {avg_chinese_m:.4f}")
print(f"  Average Effect (Native-English): {avg_native_m - avg_english_m:.4f}")
print(f"  Average Effect (Native-Chinese): {avg_native_m - avg_chinese_m:.4f}")
print(f"  Statistically Significant Effects (p < 0.05, Native vs English): {percent_sig_eng_m:.2f}% ({sig_effects_eng_m}/{total_m})")
print(f"  Statistically Significant Effects (p < 0.05, Native vs Chinese): {percent_sig_zh_m:.2f}% ({sig_effects_zh_m}/{total_m})")

# Aggregate t-tests for feminine grammar
print("\n\n===== AGGREGATED STATISTICAL TESTS: FEMININE GRAMMAR =====")
fem_df = summary_df[summary_df['Grammar Gender'] == 'female']

# Count significant effects
sig_effects_eng_f = 0
sig_effects_zh_f = 0
total_f = 0

for lang in fem_df['Language'].unique():
    for model in fem_df[fem_df['Language'] == lang]['Model'].unique():
        t_test_result = perform_t_test(original_df, lang, model, 'female')
        if not np.isnan(t_test_result['p_value_eng']):
            total_f += 1
            if t_test_result['p_value_eng'] < 0.05:
                sig_effects_eng_f += 1
            
            if t_test_result['p_value_zh'] < 0.05:
                sig_effects_zh_f += 1

percent_sig_eng_f = (sig_effects_eng_f / total_f) * 100 if total_f > 0 else 0
percent_sig_zh_f = (sig_effects_zh_f / total_f) * 100 if total_f > 0 else 0

# Calculate average values
avg_native_f = fem_df['Native Female Bias'].mean()
avg_english_f = fem_df['English Female Bias'].mean()
avg_chinese_f = fem_df['Chinese Female Bias'].mean()

print(f"Average Female Representation:")
print(f"  Native: {avg_native_f:.4f}, English: {avg_english_f:.4f}, Chinese: {avg_chinese_f:.4f}")
print(f"  Average Effect (Native-English): {avg_native_f - avg_english_f:.4f}")
print(f"  Average Effect (Native-Chinese): {avg_native_f - avg_chinese_f:.4f}")
print(f"  Statistically Significant Effects (p < 0.05, Native vs English): {percent_sig_eng_f:.2f}% ({sig_effects_eng_f}/{total_f})")
print(f"  Statistically Significant Effects (p < 0.05, Native vs Chinese): {percent_sig_zh_f:.2f}% ({sig_effects_zh_f}/{total_f})")

# Overall summary
print("\n\n===== OVERALL STATISTICAL SUMMARY =====")
total_combinations = total_m + total_f
total_sig_eng = sig_effects_eng_m + sig_effects_eng_f
total_sig_zh = sig_effects_zh_m + sig_effects_zh_f

percent_total_sig_eng = (total_sig_eng / total_combinations) * 100 if total_combinations > 0 else 0
percent_total_sig_zh = (total_sig_zh / total_combinations) * 100 if total_combinations > 0 else 0

print(f"Total Statistically Significant Effects (p < 0.05, Native vs English): {percent_total_sig_eng:.2f}% ({total_sig_eng}/{total_combinations})")
print(f"Total Statistically Significant Effects (p < 0.05, Native vs Chinese): {percent_total_sig_zh:.2f}% ({total_sig_zh}/{total_combinations})")

# Create a summary visualization
print("\n\nGenerating visualizations...")

plt.figure(figsize=(15, 8))

# Average values by grammar gender
masc_df = summary_df[summary_df['Grammar Gender'] == 'male']
fem_df = summary_df[summary_df['Grammar Gender'] == 'female']

# Calculate means for presentation
masc_native_mean = masc_df['Native Male Bias'].mean()
masc_english_mean = masc_df['English Male Bias'].mean()
masc_chinese_mean = masc_df['Chinese Male Bias'].mean()

fem_native_mean = fem_df['Native Female Bias'].mean()
fem_english_mean = fem_df['English Female Bias'].mean()
fem_chinese_mean = fem_df['Chinese Female Bias'].mean()

# Data for plotting
x = np.array([0])
width = 0.25

# Plot the data
plt.subplot(1, 2, 1)
plt.bar(x - width, masc_native_mean, width, label='Native', color='blue')
plt.bar(x, masc_english_mean, width, label='English', color='orange')
plt.bar(x + width, masc_chinese_mean, width, label='Chinese', color='green')
plt.ylabel('Male Representation')
plt.title(f'Effect of Masculine Grammar\n{percent_sig_eng_m:.0f}% significant vs English, {percent_sig_zh_m:.0f}% vs Chinese')
plt.xticks(x, ['Masculine Grammar'])
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(x - width, fem_native_mean, width, label='Native', color='blue')
plt.bar(x, fem_english_mean, width, label='English', color='orange')
plt.bar(x + width, fem_chinese_mean, width, label='Chinese', color='green')
plt.ylabel('Female Representation')
plt.title(f'Effect of Feminine Grammar\n{percent_sig_eng_f:.0f}% significant vs English, {percent_sig_zh_f:.0f}% vs Chinese')
plt.xticks(x, ['Feminine Grammar'])
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/t-test/grammar_gender_effect_with_stats.png', dpi=300)
print("Saved visualization to 'grammar_gender_effect_with_stats.png'")

# Save full statistical results to CSV
statistical_results = []

for lang in summary_df['Language'].unique():
    for gender in ['male', 'female']:
        gender_df = summary_df[(summary_df['Language'] == lang) & 
                              (summary_df['Grammar Gender'] == gender)]
        
        for _, row in gender_df.iterrows():
            model = row['Model']
            
            if gender == 'male':
                native_value = row['Native Male Bias']
                english_value = row['English Male Bias']
                chinese_value = row['Chinese Male Bias']
                comparison_type = "Male Representation"
            else:
                native_value = row['Native Female Bias']
                english_value = row['English Female Bias']
                chinese_value = row['Chinese Female Bias']
                comparison_type = "Female Representation"
            
            # Calculate differences
            native_eng_diff = native_value - english_value
            native_zh_diff = native_value - chinese_value
            
            # Perform t-test
            t_test_result = perform_t_test(original_df, lang, model, gender)
            
            statistical_results.append({
                'Language': lang,
                'Grammar Gender': gender,
                'Model': model,
                'Comparison Type': comparison_type,
                'Native Value': native_value,
                'English Value': english_value,
                'Chinese Value': chinese_value,
                'Effect (Native-English)': native_eng_diff,
                'Effect (Native-Chinese)': native_zh_diff,
                't-statistic (Native-English)': t_test_result['t_stat_eng'],
                'p-value (Native-English)': t_test_result['p_value_eng'],
                't-statistic (Native-Chinese)': t_test_result['t_stat_zh'],
                'p-value (Native-Chinese)': t_test_result['p_value_zh'],
                'Sample Size': t_test_result['sample_size'],
                'Significant (Native-English)': np.nan if np.isnan(t_test_result['p_value_eng']) else t_test_result['p_value_eng'] < 0.05,
                'Significant (Native-Chinese)': np.nan if np.isnan(t_test_result['p_value_zh']) else t_test_result['p_value_zh'] < 0.05
            })

stats_df = pd.DataFrame(statistical_results)
stats_df.to_csv('./data UAI/MutliModalSocialBias/Grammar is Bias ?/Classification/reportsFull/t-test/statistical_tests_results.csv', index=False)
print("Saved statistical test results to 'statistical_tests_results.csv'")
