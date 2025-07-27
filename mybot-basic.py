#!/usr/bin/env python3

# === AIML + Similarity-based Chatbot ===

import aiml
import wikipedia
import pandas as pd
import sys
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import requests

# Setup NLTK
nltk.download('punkt')
nltk.download('wordnet')
read_expr = Expression.fromstring
lemmatizer = WordNetLemmatizer()

API_NINJAS_KEY = "ANl5H3TftYAuQL/RTuVH6g==TTM2mW9LNsQ0Txj6"

# === Animal API Functions ===
def get_animal_info(animal_name):
    url = f"https://api.api-ninjas.com/v1/animals?name={animal_name}"
    headers = {"X-Api-Key": API_NINJAS_KEY}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data:
            return {"status": "not_found", "message": f"No data found for '{animal_name}'."}
        return {"status": "success", "animal": data[0]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def format_api_response(cmd, animal_name, animal_data):
    char = animal_data.get("characteristics", {})
    tax = animal_data.get("taxonomy", {})
    animal_name = animal_name.capitalize()

    if cmd == 2:
        return f"The scientific name of the {animal_name} is {tax.get('scientific_name', 'not available')}."
    elif cmd == 3:
        return f"{animal_name}s typically live for {char.get('lifespan', 'an unknown number of')} years."
    elif cmd == 4:
        locations = ", ".join(animal_data.get("locations", ["unknown locations"]))
        return f"{animal_name}s are commonly found in {locations}."
    elif cmd == 5:
        return f"{animal_name}s eat {char.get('diet', 'unknown food')}."
    elif cmd == 6:
        return f"The top speed of a {animal_name} is {char.get('top_speed', 'unknown')}."
    elif cmd == 7:
        return f"A baby {animal_name} is called a {char.get('name_of_young', 'unknown')}."
    elif cmd == 8:
        return f"A {animal_name} typically weighs {char.get('weight', 'an unknown amount')}."
    elif cmd == 9:
        return f"The most distinctive feature of a {animal_name} is {char.get('most_distinctive_feature', 'unknown')}."
    elif cmd == 10:
        return f"The gestation period of a {animal_name} is {char.get('gestation_period', 'unknown')}."
    elif cmd == 11:
        return f"{animal_name}s usually exhibit {char.get('group_behavior', 'unknown')} behavior."
    elif cmd == 12:
        return f"A {animal_name} belongs to the class {tax.get('class', 'unknown')}."
    else:
        return "I have some information, but I'm not sure how to present it!"

# === Lemmatization ===
def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

# === Knowledge Base ===
def check_kb_integrity(kb):
    for i, expr1 in enumerate(kb):
        for j, expr2 in enumerate(kb):
            if i != j and ResolutionProver().prove(expr1.negate(), [expr2]):
                raise ValueError(f"Contradiction found between: {expr1} and {expr2}")

kb = []
data = pd.read_csv(r"C:\Users\sunad\Downloads\logical-kb.csv", header=None, encoding='utf-8')
[kb.append(read_expr(row.upper())) for row in data[0]]
try:
    check_kb_integrity(kb)
    print("Knowledge base is consistent and has no contradictions.")
except ValueError as e:
    print(f"Error in KB: {e}")
    exit()

# === Load and Process QA CSV ===
qa_data = pd.read_csv(r"C:\Users\sunad\OneDrive\Belgeler\AI Lab Tasks\Animals_Chatbot_QA.csv")
qa_data['Lemmatized_Question'] = qa_data['Question'].apply(lemmatize_text)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(qa_data['Lemmatized_Question'])

# === Similarity-based matching ===
def find_most_similar_question(user_query, tfidf_matrix, vectorizer, qa_data):
    lemmatized_query = lemmatize_text(user_query)
    query_vector = vectorizer.transform([lemmatized_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    most_similar_idx = similarities.argmax()
    max_similarity = similarities[0][most_similar_idx]
    return qa_data.iloc[most_similar_idx]['Question'], qa_data.iloc[most_similar_idx]['Answer'], max_similarity

# === Logic-Based Commands ===
def cmd_31_add_fact(params, kb):
    try:
        object_raw, subject_raw = params.lower().split(' is ')
        object_clean = object_raw.strip().capitalize()
        subject_clean = subject_raw.strip().replace("a ", "").replace("an ", "").capitalize()

        new_expr = read_expr(f"{subject_clean}({object_clean})")

        # Check for contradiction using the negation of the new fact
        contradiction = ResolutionProver().prove(new_expr.negate(), kb)
        if contradiction:
            print(f"Error: Adding '{new_expr}' causes a contradiction with the current knowledge base.")
            return

        temp_kb = kb + [new_expr]
        check_kb_integrity(temp_kb)
        kb.append(new_expr)
        print(f"OK, I will remember that {object_clean} is {subject_clean}.")
    except Exception as e:
        print(f"Could not add fact due to error: {e}")


def cmd_32_check_fact(params, kb):
    try:
        object_raw, subject_raw = params.lower().split(' is ')
        object_clean = object_raw.strip().capitalize()
        subject_clean = subject_raw.strip().replace("a ", "").replace("an ", "").capitalize()
        expr = read_expr(f"{subject_clean}({object_clean})")
        result = ResolutionProver().prove(expr, kb)
        if result:
            print("Correct.")
        elif ResolutionProver().prove(expr.negate(), kb):
            print("Incorrect.")
        else:
            print("Sorry, I don't know.")
    except Exception as e:
        print(f"Could not check fact due to error: {e}")

# === AIML Kernel ===
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# === Extract Animal Name ===
def extract_animal(text):
    words = word_tokenize(text.lower())
    for word in words[::-1]:
        if word not in {
            "what", "do", "does", "is", "are", "the", "a", "an", "of",
            "how", "many", "much", "eat", "drink", "run", "swim", "called",
            "speed", "kind", "name", "live", "lives", "weigh", "feature",
            "gestation", "period", "behavior", "group", "fast"
        }:
            return lemmatizer.lemmatize(word)
    return None

print("Welcome to this chat bot. Please feel free to ask questions from me!")

# === Main Chat Loop ===
while True:
    try:
        user_input = input(">> ").strip()
        if not user_input:
            continue

        answer = kern.respond(user_input)

        if answer.startswith('#'):
            params = answer[1:].split('$')
            cmd = int(params[0])
            animal = params[1].strip().lower()

            if cmd == 31:
                cmd_31_add_fact(animal, kb)
                continue
            elif cmd == 32:
                cmd_32_check_fact(animal, kb)
                continue

            if animal.startswith("a "):
                animal = animal[2:]
            elif animal.startswith("an "):
                animal = animal[3:]
            animal = lemmatizer.lemmatize(animal)

            matched_q, qa_answer, sim = find_most_similar_question(user_input, tfidf_matrix, vectorizer, qa_data)
            matched_animal = extract_animal(matched_q)
            user_animal = extract_animal(user_input)

            if sim >= 0.85 and matched_animal == user_animal:
                print("Answer:", qa_answer)
                print(f"(Matched with {sim:.2f} similarity)")
                continue

            response = get_animal_info(animal)
            if response["status"] == "success":
                formatted = format_api_response(cmd, animal, response["animal"])
                print(formatted)
            elif response["status"] == "not_found":
                print(f"Sorry, I couldn’t find any data for '{animal}'.")
            else:
                print(f"API Error: {response['message']}")

        elif answer:
            print(answer)
        else:
            _, answer, sim = find_most_similar_question(user_input, tfidf_matrix, vectorizer, qa_data)
            print("Answer:", answer)
            if sim:
                print(f"(Matched with {sim:.2f} similarity)")

    except (KeyboardInterrupt, EOFError):
        print("Goodbye!")
        break
