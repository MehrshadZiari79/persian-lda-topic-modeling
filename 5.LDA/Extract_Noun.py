import dadmatools.pipeline.language as language

# Initialize the DadmaTools pipeline with lemmatizer, POS tagger, and other necessary tools
pips = 'tok, lem, pos, dep, chunk, cons, spellchecker, kasreh, itf, ner, sent'
nlp = language.Pipeline(pips)

# Sample Persian sentence
text = "کشور بزرگ ایران توانسته در طی سال‌ها اغشار مختلفی از قومیت‌های گوناگون رو به خوبی تو خودش جا بده"

# Process the text using the pipeline
doc = nlp(text)

# Extract Noun-Adjective pairs
noun_adj_pairs = []

# Iterate through sentences and tokens to identify noun-adjective pairs
for sentence in doc['sentences']:
    tokens = sentence['tokens']
    for i, token in enumerate(tokens):
        if token['upos'] == 'ADJ':  # Check if the token is an adjective
            if i > 0 and tokens[i-1]['upos'] == 'NOUN':  # Check if the previous token is a noun
                noun_adj_pairs.append((tokens[i-1]['text'], token['text']))

# Print the extracted noun-adjective pairs
print(noun_adj_pairs)
