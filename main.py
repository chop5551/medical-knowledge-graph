# main.py

import os
import json
import openai
import networkx as nx
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to import the ontology from a JSON file
def import_ontology(ontology_path):
    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
    return ontology

# Function to load the regulatory medical text
def load_medical_text(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        medical_text = f.read()
    return medical_text

# Function to generate the prompt for GPT
def generate_prompt(medical_text, ontology):
    prompt = (
        "As a medical expert, please analyze the following medical text and extract information about "
        "diseases, symptoms, treatment methods, used medications, dosages, and contraindications. "
        "Structure the results in JSON format according to the given ontology.\n\n"
        f"Ontology:\n{json.dumps(ontology, ensure_ascii=False, indent=2)}\n\n"
        f"Text:\n{medical_text}\n\n"
        "Answer:"
    )
    return prompt

# Function to call the GPT model and get structured data
def extract_information(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical expert assisting in extracting information from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.2,
    )
    extracted_data = response.choices[0].message['content']
    return extracted_data

# Function to build the knowledge graph based on extracted data
def build_knowledge_graph(extracted_data):
    data = json.loads(extracted_data)
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    disease = data.get('Disease', 'Unknown')
    G.add_node(disease, type='Disease')

    symptoms = data.get('Symptoms', [])
    for symptom in symptoms:
        G.add_node(symptom, type='Symptom')
        G.add_edge(disease, symptom, relation='has_symptom')

    treatment = data.get('Treatment', {})
    if treatment:
        drug = treatment.get('Medication', 'Unknown')
        G.add_node(drug, type='Medication')
        G.add_edge(disease, drug, relation='treated_with')

        dosage = treatment.get('Dosage', {})
        for group, dose in dosage.items():
            group_node = f"{drug}_{group}"
            G.add_node(group_node, type='Dosage')
            G.add_edge(drug, group_node, relation='recommended_for')
            G.nodes[group_node]['dosage'] = dose

    contraindications = treatment.get('Contraindications', [])
    for contraindication in contraindications:
        G.add_node(contraindication, type='Contraindication')
        G.add_edge(drug, contraindication, relation='has_contraindication')

    return G

# Function to export the knowledge graph to a JSON file
def export_knowledge_graph(G, output_path):
    data = nx.readwrite.json_graph.node_link_data(G)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Main function
def main():
    # File paths (replace with your paths)
    ontology_path = 'ontology.json'
    medical_text_path = 'medical_text.txt'
    output_graph_path = 'knowledge_graph.json'

    # Step 1: Import the ontology
    ontology = import_ontology(ontology_path)

    # Step 2: Load the medical text
    medical_text = load_medical_text(medical_text_path)

    # Step 3: Generate the prompt and extract information
    prompt = generate_prompt(medical_text, ontology)
    extracted_data = extract_information(prompt)

    # Output the extracted data
    print("Extracted Data:")
    print(extracted_data)

    # Step 4: Build the knowledge graph
    G = build_knowledge_graph(extracted_data)

    # Step 5: Export the knowledge graph
    export_knowledge_graph(G, output_graph_path)
    print(f"Knowledge graph successfully saved to {output_graph_path}")

if __name__ == '__main__':
    main()
