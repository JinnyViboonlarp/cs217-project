import torch
import sys
import io

from model import DANTextClassifier
from torch.utils.data import DataLoader

#========== Do the necessary loading and setting things up ================

max_len = 16
label_map = {"O": 0, "NAME": 1, "STATE": 2, "UNIT": 3, "QUANTITY": 4,
                 "SIZE": 5, "TEMP": 6, "DF": 7}
label_map_inv = {v: k for k, v in label_map.items()}
vocab_path = 'vocab.txt'

model = torch.load('model.pt')

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r') as fh_in:
        lines = fh_in.readlines()
    for line in enumerate(lines):
        tokens = line[1].strip().split()
        if(len(tokens) >= 2):
            (i, word) = (tokens[0], " ".join(tokens[1:]))
            vocab[word] = int(i)
        else:
            print(tokens)
    return vocab

vocab = load_vocab(vocab_path)
out_of_vocab_index = (len(vocab)+1)
padding_index = (len(vocab)+2)

def text_pipeline(words):
    return [vocab.get(word, out_of_vocab_index) for word in words]

#==========================================================================

class Entity:
    def __init__(self, id, start_char, end_char, text, label):
        self.id = id
        self.start_char = start_char
        self.end_char = end_char
        self.text = text
        self.label = label

class NER_Document:
    
    def __init__(self, text: str):
        self.text, self.entities = ner(text)
        
    def get_entities_with_markup(self):
        showed_entities = self.create_entities_list_for_markup()
        starts = {e.start_char: e.label for e in showed_entities if e.label != 'O'}
        ends = {e.end_char: True for e in showed_entities if e.label != 'O'}
        buffer = io.StringIO()
        for p, char in enumerate(self.text):
            if p in ends:
                buffer.write('</entity>')
            if p in starts:
                buffer.write('<entity class="%s">' % starts[p])
            buffer.write(char)
        markup = buffer.getvalue()
        return '<markup>%s</markup>' % markup

    def create_entities_list_for_markup(self):
        real_entities = self.entities
        showed_entities = []
        i = 0
        while(i < len(real_entities)): 
            entity = Entity(real_entities[i].id , real_entities[i].start_char, real_entities[i].end_char,
                            real_entities[i].text, real_entities[i].label) #can't copy normally
            while((i+1)<len(real_entities) and real_entities[i+1].label ==  entity.label):
                entity.end_char = real_entities[i+1].end_char
                i += 1
            showed_entities.append(entity)
            i += 1
        return showed_entities
            

def tokenize(text):
    text = text.replace(',',' , ').replace('(',' ( ').replace(')',' ) ')
    return text.strip().split()[:max_len]
    #return text.replace(',',' , ').strip().split()[:max_len]

def get_loader(tokenized_text, batch_size=32, max_len=16, shuffle=False):

    def collate_batch(batch):

        text_list, mask_list = [], []
        pre_augmented_text = text_pipeline(tokenized_text)
        print(pre_augmented_text)
        for i,w in enumerate(pre_augmented_text):
            text = [w] + pre_augmented_text
            # If too short, pad to max_len.
            text = text + ([padding_index] * max_len)
            # If sentence too long, truncate to max_len.
            text = text[:max_len]
            # Create a mask tensor indicating where padding is present.
            pad_mask = [0 if i==padding_index else 1 for i in text]

            text_list.append(text)
            mask_list.append(pad_mask)

        text_list = torch.tensor(text_list)
        mask_list = torch.tensor(mask_list)
        return text_list, mask_list
    
    return DataLoader(tokenized_text, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch)

def predict(model, dataloader, label_map_inv):
    model.eval() # do not use dropout
    with torch.no_grad():
        # Iterate over the data in the data loader
        pred = []
        for i, (ids, mask) in enumerate(dataloader):
            pass
            # Perform forward pass to get outputs (like in train)
            output = model(ids, mask)
            # Get the predictions by taking the argmax
            int_pred = torch.argmax(output, dim=1).tolist()
            pred = pred + [label_map_inv[w] for w in int_pred]
  
    return pred
    # e.g. ['QUANTITY', 'SIZE', 'NAME', 'NAME', 'NAME', 'O', 'O', 'STATE']

def create_entity_list(tokenized_text, pred):

    def valid(word):
        if(word in [',','(',')']):
            return False
        return True

    new_text = " ".join(tokenized_text)
    entities = []
    start_char = 0
    i = 0
    while(i < len(pred)):
        word = tokenized_text[i]
        tag = pred[i]
        end_char = (start_char + len(word))
        if(not(valid(word))):
            tag = 'O'
        entity = Entity(i, start_char, end_char, word, tag)  # i is used as the unique id
        entities.append(entity)
        start_char = (end_char + 1)
        i += 1
    return new_text, entities 

def ner(text):

    tokenized_text = tokenize(text)
    dataloader = get_loader(tokenized_text, batch_size=32, max_len=max_len, shuffle=False)
    pred = predict(model, dataloader, label_map_inv)
    new_text, entities = create_entity_list(tokenized_text, pred)
    return new_text, entities
    
if __name__ == "__main__":

    new_text, entities = ner("1/2 large sweet red onion, thinly sliced")
    print(new_text)
