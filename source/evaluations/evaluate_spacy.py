import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_pytt_distilbertbaseuncased_lg")

doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion unsuspecting")
for token in doc:
    print(f"{token.text}, {token.pos_}, {token.dep_}, {len(token.vector)}")