import ast
import json
import pandas as pd  
from scipy import spatial  
from openai import OpenAI

from utils.helpers import calculate_tokens



import ast
import pandas as pd
from scipy import spatial
from openai import OpenAI

class BindingDBSearch:
    def __init__(self, config):
        self.client = OpenAI()
        self.gpt_model=config.GPT_MODEL
        self.token_budget=config.TOKEN_BUDGET
        self.embedding_model=config.EMBEDDING_MODEL
        
        self._query_message_added = False
        self.messages = [{"role": "system", "content": "You answer the user's question by incorporating relevant BindingDB results provided to you in the context."},]
        
        self.df = pd.read_csv(config.csv_with_embeddings_path)
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    def strings_ranked_by_relatedness(
        self,
        query: str,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 5
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [
            (row['combined'], relatedness_fn(query_embedding, row["embedding"]))
            for idx, row in self.df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def _print_binding_entry(self, st, reltd, idx):
        fields = [field.strip() for field in st.split(',')]
        formatted_fields = '\n    '.join(fields)
        
        entry_str = f"""
    🔍 Result #{idx + 1} (Relevance: {reltd:.3f})
    ----------------------------------------
        {formatted_fields}
    """
        print(entry_str)

    def _query_message(self, query: str) -> str:
        """Constructs the query message with relevant context."""
        strings, relatednesses = self.strings_ranked_by_relatedness(query)
        introduction = 'Use the below entires from BindingDB to answer the user question: '
        question = f"\n\nQuestion: {query}"
        message = introduction
        
        for idx in range(len(strings)):
            st, reltd = strings[idx], relatednesses[idx]
            next_article = f'\nBindingDB item number {idx} with relatedness score of {reltd:.3f}: The row content json.dumped string: \n"""\n{st}\n"""'
            if (
                calculate_tokens(message + next_article + question, model=self.gpt_model)
                > self.token_budget
            ):
                break
            else:
                message += next_article
                self._print_binding_entry(st=next_article, reltd=reltd, idx=idx)
        self._query_message_added = True
        return message + question


    def ask(self, message: str) -> str:
        """Answers a query using GPT and the dataframe of relevant texts and embeddings."""
        if not self._query_message_added:
            message = self._query_message(message)
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=self.messages,
            temperature=0
        )
        response_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_content})
        return response_content


def chat_interface(bdb):
    print("🤖 Welcome to BindingDB Search Assistant! Type 'exit' to quit.\n")
    
    while True:
        user_query = input("👤 You: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("\n🤖 Goodbye! Have a great day! 👋")
            break
            
        if not user_query:
            continue
            
        print("🤖 ASSISTANT ANALYSIS:")
        print("-"*80)
        response = bdb.ask(user_query)
        print(response)
        print("-"*80 + "\n")

if __name__ == "__main__":
    from configs import SMALL_NO_NA_NO_CHIANS_CONFIG as config
    bdb = BindingDBSearch(config=config)
    chat_interface(bdb=bdb)