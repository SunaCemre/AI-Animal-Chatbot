#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import pandas as pd
import requests
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
import re
import csv

import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np


# Load pre-trained MobileNetV2 model
mobilenet_model = load_model("animal_classifier_mobilenetv2.keras", compile=False)
class_labels = ['cats', 'dogs', 'snakes']  


kb_file = "logical_kb.csv"

# Helper function to load the knowledge base
def load_kb():
    try:
        df = pd.read_csv(kb_file, names=['fact'], header=None)
        df['fact'] = df['fact'].astype(str).str.strip().str.lower()
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['fact'])

# Helper function to check if a fact is in valid logical syntax
def is_valid_logical_fact(fact):
    import re
    # Supports predicate(x), predicate(x) -> predicate(y), etc.
    return bool(re.match(r'^([a-z_]+\([^\(\)]*\)|[a-z_]+\([^\(\)]*\)\s*->\s*-?[a-z_]+\([^\(\)]*\))$', fact))

# Final save function
def save_fact_to_kb(fact):
    fact = fact.strip().lower()
    kb_df = load_kb()

    if not is_valid_logical_fact(fact):
        print(f"[WARNING] Not a valid logical fact. Skipping: {fact}")
        return

    if fact in kb_df['fact'].values:
        print(f"[INFO] Fact '{fact}' already exists. Not saving duplicate.")
        return

    with open(kb_file, mode="a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([fact])
        print(f"[SUCCESS] Saved to KB: {fact}")



#nltk.download('punkt')
#nltk.download('wordnet')
read_expr = Expression.fromstring
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
# API setup
API_NINJAS_KEY = "ANl5H3TftYAuQL/RTuVH6g==TTM2mW9LNsQ0Txj6"


def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]
    return ' '.join(lemmatized)

# Load Question-Answer CSV into memory
def load_qa_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df['question_lemmatized'] = df['question'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x)))
    return df, vectorizer.fit_transform(df['question_lemmatized'])

# Load once near the top:
qa_df, tfidf_matrix = load_qa_csv("Animals_Chatbot_QA.csv")



def get_best_answer(user_input, df, vectorizer, tfidf_matrix):
    lemmatized_input = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(user_input))
    query_vector = vectorizer.transform([lemmatized_input])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_idx = cosine_similarities.argmax()
    max_similarity = cosine_similarities[best_idx]
    
    print(f"Debug: Best match score is {max_similarity}")  # Debug line to check similarity score
    
    if max_similarity > 0.3:  # Check if the similarity threshold is appropriate
        return df.iloc[best_idx]['answer']
    else:
        return None


        
        
def check_contradiction(new_fact, filepath):

    df = load_kb()
    new_fact_clean = new_fact.strip().lower()

    # Parse the new predicate and subject
    match = re.match(r"^([a-z_]+)\(([^()]+)\)$", new_fact_clean)
    if not match:
        return False  # If it's not in predicate(subject) format, can't reason about contradiction

    new_predicate, new_subject = match.groups()

    # Loop through KB and look for same subject with different predicate
    for fact in df['fact'].dropna().str.strip().str.lower():
        match_existing = re.match(r"^([a-z_]+)\(([^()]+)\)$", fact)
        if match_existing:
            existing_predicate, existing_subject = match_existing.groups()
            if existing_subject == new_subject and existing_predicate != new_predicate:
                # Found same subject with a different predicate
                return True

    return False



def convert_to_fol(fact_str):
    # Remove articles and lowercase everything
    fact_str = fact_str.lower().replace(" a ", " ").replace(" an ", " ").replace(" the ", " ").strip()

    # Expect format like: "robin is bird" or "polar bear is mammal"
    match = re.match(r"^(.+?)\s+is\s+(.+?)$", fact_str)
    if match:
        subject = match.group(1).strip().replace(" ", "")
        category = match.group(2).strip().replace(" ", "")
        return f"{category}({subject})"
    else:
        return None


def process_logical_input(user_input, kb_filepath):
    user_input = user_input.strip().lower()

    # "I know that X is Y" or "I know that X has Y"
    if user_input.startswith("i know that"):
        fact_part = user_input[len("i know that"):].strip()
        if " is " in fact_part:
            subject, predicate = [x.strip().replace(" ", "") for x in fact_part.split(" is ", 1)]
            fact = f"{predicate}({subject})"
        elif " has " in fact_part:
            subject, attribute = [x.strip().replace(" ", "") for x in fact_part.split(" has ", 1)]
            fact = f"has{attribute}({subject})"
        else:
            return "I understand what you said, but it's not a valid logical format I can store."

        # First check for exact match
        if fact in load_kb()['fact'].values:
            return f"I already know that {fact_part}."
        # Then check for contradiction
        elif check_contradiction(fact, kb_filepath):
            return f"That contradicts what I already know about {subject}."
        else:
            save_fact_to_kb(fact)
            return f"OK, I will remember that {fact_part}."

    # "Check that X is Y" or "Check that X has Y"
    elif user_input.startswith("check that"):
        fact_part = user_input[len("check that"):].strip()

        if " is " in fact_part:
            subject, predicate = [x.strip().replace(" ", "") for x in fact_part.split(" is ", 1)]
            fact = f"{predicate}({subject})"
        elif " has " in fact_part:
            subject, attribute = [x.strip().replace(" ", "") for x in fact_part.split(" has ", 1)]
            fact = f"has{attribute}({subject})"
        else:
            return "I understand what you said, but it's not a valid logical format I can check."

        # Load current KB facts
        existing_facts = load_kb()['fact'].dropna().str.strip().str.lower()

        if fact in existing_facts.values:
            return "Correct"
        elif check_contradiction(fact, kb_filepath):
            return f"No, that's incorrect. It contradicts what I know about {subject}."
        else:
            return "I don’t know."

    return None




# Function to fetch Wikipedia summary
def get_wikipedia_summary(topic):
    try:
        summary = wikipedia.summary(topic, sentences=2, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple entries found for {topic}, please be more specific."
    except wikipedia.exceptions.PageError:
        return "No information found on Wikipedia."


def handle_command_response(response):
    if response.startswith("#"):
        command, param = response[1:].split("$", 1)
        if command == "2":
            return get_scientific_name(param)
        elif command == "99":
            return "Handling generic inquiry for: " + param
        else:
            return "Unhandled command"
    else:
        return response

def get_scientific_name(animal_name):
    animal_info = get_animal_info(animal_name)
    if animal_info:
        return f"The scientific name of {animal_name} is {animal_info.get('scientific_name', 'Unknown')}."
    else:
        return f"Scientific name for {animal_name} not found."
    
def format_api_response(cmd, animal_name, animal_data):
    """
    Format the API response based on the command number and data received from the API.
    
    Args:
    - cmd (int): The command number indicating what type of information is requested.
    - animal_name (str): The name of the animal.
    - animal_data (dict): The data dictionary returned from the API.
    
    Returns:
    - str: Formatted string response based on the command.
    """
    # Accessing characteristics and taxonomy from the animal data
    char = animal_data.get("characteristics", {})
    tax = animal_data.get("taxonomy", {})

    # Capitalize the first letter of the animal name for presentation
    animal_name = animal_name.capitalize()

    if cmd == 2:
        return f"The scientific name of {animal_name} is {tax.get('scientific_name', 'information unavailable')}."
    elif cmd == 3:
        return f"{animal_name}s typically live for about {char.get('lifespan', 'a variable number of')} years."
    elif cmd == 4:
        locations = ", ".join(animal_data.get("locations", ["specific locations not available"]))
        return f"{animal_name}s are commonly found in {locations}."
    elif cmd == 5:
        return f"{animal_name}s primarily are {char.get('diet', 'various foods')}."
    elif cmd == 6:
        return f"The top speed of a {animal_name} is {char.get('top_speed', 'speed data not available')}."
    elif cmd == 7:
        return f"A baby {animal_name} is called a {char.get('name_of_young', 'specific term not available')}."
    elif cmd == 8:
        return f"A typical {animal_name} weighs {char.get('weight', 'weight data unavailable')}."
    elif cmd == 9:
        return f"The most distinctive feature of a {animal_name} is {char.get('most_distinctive_feature', 'distinctive feature not specified')}."
    elif cmd == 10:
        return f"The gestation period for a {animal_name} is about {char.get('gestation_period', 'gestation period data unavailable')}."
    elif cmd == 11:
        return f"{animal_name}s typically exhibit {char.get('group_behavior', 'group behavior details unavailable')} behavior."
    elif cmd == 12:
        return f"{animal_name} belongs to the {tax.get('class', 'class information unavailable')} class."

    return "Information requested is unavailable."



def classify_image_from_dialog():
    try:
        root = tk.Tk()
        root.lift()
        root.attributes('-topmost', True)
        root.withdraw()
        
        file_path = filedialog.askopenfilename(title="Select an image")

        if not file_path:
            return "No image selected."

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        pred = mobilenet_model.predict(img_array)
        predicted_index = np.argmax(pred)
        confidence = pred[0][predicted_index]

        return f"That looks like a {class_labels[predicted_index]} with {confidence:.0%} confidence."
    except Exception as e:
        return f"An error occurred during image classification: {str(e)}"



    

# Initialize AIML kernel
kern = aiml.Kernel()
kern.bootstrap(learnFiles="mybot-basic.xml")

print("Loading AIML patterns...")
kern.bootstrap(learnFiles="mybot-basic.xml")
print("AIML patterns loaded.")

# Set up NLP tools
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

# Load QA data

qa_df['question_lemmatized'] = qa_df['question'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x)))
tfidf_matrix = vectorizer.fit_transform(qa_df['question_lemmatized'])

# API setup
API_NINJAS_KEY = "ANl5H3TftYAuQL/RTuVH6g==TTM2mW9LNsQ0Txj6"
base_url = "https://api.api-ninjas.com/v1/animals?name="

# Function to get animal information from the API
def get_animal_info(animal_name):
    headers = {"X-Api-Key": API_NINJAS_KEY}
    response = requests.get(f"{base_url}{animal_name}", headers=headers)
    if response.status_code == 200:
        #print(response.json())
        return response.json()[0]
    
    return None


# Load QA data
qa_df = pd.read_csv("Animals_Chatbot_QA.csv")
qa_df.columns = qa_df.columns.str.strip().str.lower()  # normalize column names

qa_df['question_lemmatized'] = qa_df['question'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x)))
tfidf_matrix = vectorizer.fit_transform(qa_df['question_lemmatized'])

# Initialize AIML kernel
kern = aiml.Kernel()
kern.bootstrap(learnFiles="mybot-basic.xml")

print("Welcome to this chatbot. Please feel free to ask me questions!")

while True:
    try:
        user_input = input(">> ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # 1. Logical KB handling
        kb_result = process_logical_input(user_input, "logical_kb.csv")
        if kb_result:
            print(kb_result)
            continue

        # 2. AIML response
        response = kern.respond(user_input)

        # 3. Image classification
        if response == "#image_classification":
            print(classify_image_from_dialog())
            continue

        # 4. Handle AIML command responses like #1$elephant
        if response.startswith("#"):
            parts = response[1:].split("$", 1)
            cmd = int(parts[0])
            param = parts[1].strip()

            if cmd == 1:
                # Wikipedia command
                print(get_wikipedia_summary(param))
            elif cmd == 99:
                # Catch-all fallback → now handled using get_best_answer
                csv_answer = get_best_answer(user_input, qa_df, vectorizer, tfidf_matrix)
                if csv_answer:
                    print(csv_answer)
                else:
                    print(get_wikipedia_summary(user_input))
            else:
                # Other API-based commands
                animal_data = get_animal_info(param.lower())
                if animal_data:
                    print(format_api_response(cmd, param, animal_data))
                else:
                    # Final fallback if no data
                    csv_answer = get_best_answer(user_input, qa_df, vectorizer, tfidf_matrix)
                    if csv_answer:
                        print(csv_answer)
                    else:
                        print(f"Sorry, I couldn't find information on '{param}'. Try rephrasing.")
            continue

        # 5. If AIML gives a regular response
        if response:
            print(response)
        else:
            # 6. Fallback to CSV
            csv_answer = get_best_answer(user_input, qa_df, vectorizer, tfidf_matrix)
            if csv_answer:
                print(csv_answer)
            else:
                # 7. Final fallback: Wikipedia
                print(get_wikipedia_summary(user_input))

    except KeyboardInterrupt:
        print("\nExiting the chatbot.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")

