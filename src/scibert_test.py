import spacy

nlp = spacy.load("en_core_sci_scibert")
sentence = "Since 9/11, military service in the United States has been characterized by wartime deployments and " \
           "reintegration challenges that contribute to a context of stress for military families."
doc = nlp(sentence)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
