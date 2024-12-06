import json
import tiktoken

def _normalize_column_name(col):
    return col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(".", "_NUMBER_").replace("/", "_AND_")

def pprint_dict(d: dict):
    print(json.dumps(d, indent=2))
    
def truncate_embedding_input_to_token_limit(text, encoding, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

def calculate_tokens(s, model):
    token_encoder = tiktoken.encoding_for_model(model)
    return len(token_encoder.encode(s))