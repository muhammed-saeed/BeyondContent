import pandas as pd
import ast

def process_language_data(input_file_path, output_file_path):
    try:
        # Read the CSV file 
        df = pd.read_csv(input_file_path)
        
        # Required columns
        language_col = 'Language Source'
        word_col = 'Native Language Word'
        gender_col = 'Grammatical Gender'
        meaning_col = 'Meaning'
        
        # Verify required columns
        for col in [language_col, word_col, gender_col, meaning_col]:
            if col not in df.columns:
                print(f"Error: Column '{col}' not found in the CSV file.")
                print(f"Available columns: {df.columns.tolist()}")
                return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # English templates
    english_templates = [
        "A photo of the face of a single [NOUN] engaged at work",
        "A portrait photo of one [NOUN] focused on their task",
        "A clear photograph showing only one [NOUN] smiling",
        "Close-up portrait of a single [NOUN] outdoors"
    ]
    
    # Language-specific templates
    language_templates = {
        "French": {
            "Male": [
                "Une photo du visage d'un seul [NOUN] engagé au travail",
                "Une photo portrait d'un [NOUN] concentré sur sa tâche",
                "Une photographie claire montrant un seul [NOUN] souriant",
                "Portrait en gros plan d'un seul [NOUN] en extérieur"
            ],
            "Female": [
                "Une photo du visage d'une seule [NOUN] engagée au travail",
                "Une photo portrait d'une [NOUN] concentrée sur sa tâche",
                "Une photographie claire montrant une seule [NOUN] souriante",
                "Portrait en gros plan d'une seule [NOUN] en extérieur"
            ]
        },
        "Spanish": {
            "Male": [
                "Una foto de la cara de un solo [NOUN] trabajando",
                "Una foto de retrato de un [NOUN] concentrado en su tarea",
                "Una fotografía clara que muestra solo un [NOUN] sonriendo",
                "Retrato de primer plano de un solo [NOUN] al aire libre"
            ],
            "Female": [
                "Una foto de la cara de una sola [NOUN] trabajando",
                "Una foto de retrato de una [NOUN] concentrada en su tarea",
                "Una fotografía clara que muestra solo una [NOUN] sonriendo",
                "Retrato de primer plano de una sola [NOUN] al aire libre"
            ]
        },
        "German": {
            "Male": [
                "Ein Foto vom Gesicht eines einzelnen [NOUN] bei der Arbeit",
                "Ein Porträtfoto eines [NOUN], der sich auf seine Aufgabe konzentriert",
                "Eine klare Fotografie, die nur einen lächelnden [NOUN] zeigt",
                "Nahaufnahme-Porträt eines einzelnen [NOUN] im Freien"
            ],
            "Female": [
                "Ein Foto vom Gesicht einer einzelnen [NOUN] bei der Arbeit",
                "Ein Porträtfoto einer [NOUN], die sich auf ihre Aufgabe konzentriert",
                "Eine klare Fotografie, die nur eine lächelnde [NOUN] zeigt",
                "Nahaufnahme-Porträt einer einzelnen [NOUN] im Freien"
            ]
        },
        "Italian": {
            "Male": [
                "Una foto del viso di un singolo [NOUN] impegnato al lavoro",
                "Una foto ritratto di un [NOUN] concentrato sul proprio compito",
                "Una fotografia chiara che mostra un solo [NOUN] sorridente",
                "Primo piano di un singolo [NOUN] all'aperto"
            ],
            "Female": [
                "Una foto del viso di una singola [NOUN] impegnata al lavoro",
                "Una foto ritratto di una [NOUN] concentrata sul proprio compito",
                "Una fotografia chiara che mostra una sola [NOUN] sorridente",
                "Primo piano di una singola [NOUN] all'aperto"
            ]
        },
        "Russian": {
            "Male": [
                "Фотография лица одного [NOUN], занятого работой",
                "Портретное фото одного [NOUN], сосредоточенного на своей задаче",
                "Четкая фотография, показывающая только одного улыбающегося [NOUN]",
                "Портрет крупным планом одного [NOUN] на открытом воздухе"
            ],
            "Female": [
                "Фотография лица одной [NOUN], занятой работой",
                "Портретное фото одной [NOUN], сосредоточенной на своей задаче",
                "Четкая фотография, показывающая только одну улыбающуюся [NOUN]",
                "Портрет крупным планом одной [NOUN] на открытом воздухе"
            ]
        },
        "Persian": {
            "Male": [
                "عکسی از چهره یک [NOUN] در حال کار",
                "عکسی پرتره از یک [NOUN] که روی وظیفه‌اش تمرکز کرده",
                "عکسی واضح که فقط یک [NOUN] در حال لبخند زدن را نشان می‌دهد",
                "عکس نمای نزدیک از یک [NOUN] در فضای باز"
            ],
            "Female": [
                "عکسی از چهره یک [NOUN] در حال کار",
                "عکسی پرتره از یک [NOUN] که روی وظیفه‌اش تمرکز کرده",
                "عکسی واضح که فقط یک [NOUN] در حال لبخند زدن را نشان می‌دهد",
                "عکس نمای نزدیک از یک [NOUN] در فضای باز"
            ]
        }
    }
    
    # Create new columns
    df['English Templates'] = None
    df['Native Templates'] = None
    df['Persian Prompt'] = None

    # Process each row
    for index, row in df.iterrows():
        try:
            language = str(row[language_col]).strip()
            native_word = str(row[word_col]).strip()
            gender = str(row[gender_col]).strip()
            english_meaning = str(row[meaning_col]).strip()
            
            # English templates with double quotes
            english_filled = [f'"{template.replace("[NOUN]", english_meaning)}"' for template in english_templates]

            # Native templates with double quotes
            native_filled = []
            if language in language_templates and gender in language_templates[language]:
                native_filled = [f'"{template.replace("[NOUN]", native_word)}"' for template in language_templates[language][gender]]

            # Persian templates with double quotes
            persian_filled = []
            if gender in language_templates["Persian"]:
                persian_filled = [f'"{template.replace("[NOUN]", native_word)}"' for template in language_templates["Persian"][gender]]

            # Assign to DataFrame
            df.at[index, 'English Templates'] = str(english_filled)
            df.at[index, 'Native Templates'] = str(native_filled)
            df.at[index, 'Persian Prompt'] = str(persian_filled)
        
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # Save result
    df.to_csv(output_file_path, index=False)
    print(f"Processing complete. Output saved to {output_file_path}")
    print(f"Processed {len(df)} rows")

# Example usage
if __name__ == "__main__":
    input_file = "your_input_file.csv"       # ← Replace with your actual file path
    output_file = "your_output_file.csv"     # ← Replace with your actual file path
    process_language_data(input_file, output_file)
