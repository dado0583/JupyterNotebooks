import spacy
from spacy import displacy
import random

# nlp = spacy.load('en') #If you use a smaller model, you will need to swap out
# nlp.vocab.strings['CURRENCY'] to something that exists in the model.
nlp = spacy.load('en_core_web_lg')

class CurrencyPairPipeline(object):
    def __init__(self, nlp):
        self.label_hash = nlp.vocab.strings["CURRENCY"]
        self.regex_ = r"[USD|EUR|GBP|JPY|CAD][A-Z|a-z]{3}|[A-Z|a-z]{3}[USD|EUR|GBP|JPY|CAD]"
 
    def __call__(self, doc):
        import re
        from spacy.tokens import Span
        new_tokens = doc.ents
        for idx, token in enumerate(doc):
            #Checking if a currency pair, e.g. USDEUR, EURUSD, etc    
            if re.search(self.regex_, token.text):
                #We found a match so need to update the entities
                span = Span(doc, idx, idx+1, label=self.label_hash)
                #Spacy only supports one label per phrase, so need to conditionally replace (e.g. USDEUR may be wrongly labelled as an ORG or GPE)
                new_tokens = tuple([t for t in new_tokens if t.start != idx]) + (span,)

        ### SEGMENT FAULT
        ### IF YOU COMMENT THE LINE BELOW IT AVOIDS THE ERROR
        ### FROM WHAT I CAN TELL THE FACT THAT ENTITIES ARE ALREADY ON THE DOC BEFORE GETTING TO THE 'ner' pipeline SEEEMS TO BE CAUSING THE ISSU
        doc.ents = new_tokens
        return doc
 
ccy_pipeline = CurrencyPairPipeline(nlp)
try:
    nlp.remove_pipe(name='ccy_pipeline')
except:
    print("Couldn't remove pipe")
    
###### SEGMENT FAULT
###### Change before='ner' to last=True to avoid segment fault.
nlp.add_pipe(ccy_pipeline, name='ccy_pipeline', before='ner')

print(nlp.pipe_names)

doc = nlp("Paris is the awesome capital of France. They use the euro. The current USDEUR rate is 1.112 to exchange currencies")

for ent in doc.ents:
    print(ent)
