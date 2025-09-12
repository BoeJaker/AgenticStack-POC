# fastapi_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# Optional: relation extraction model or simple heuristic

class TextInput(BaseModel):
    text: str

@app.post("/ner")
def extract_ner(input: TextInput):
    entities = ner_pipeline(input.text)
    results = []
    for e in entities:
        results.append({
            "name": e['word'],
            "type": e['entity_group'],
            "start": e['start'],
            "end": e['end']
        })
    return results

@app.post("/relations")
def extract_relations(input: TextInput):
    # Naive heuristic: ORG + PERSON -> leads, ORG + PRODUCT -> develops
    # You can replace with a proper relation extraction model
    text = input.text.lower()
    entities = ner_pipeline(input.text)
    orgs = [e['word'] for e in entities if e['entity_group']=='ORG']
    people = [e['word'] for e in entities if e['entity_group']=='PER']
    relations = []
    for org in orgs:
        for p in people:
            if org.lower() in text and p.lower() in text:
                relations.append({"source": org, "target": p, "type": "leads"})
    return relations
