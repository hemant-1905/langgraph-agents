from typing import List, TypedDict

class GraphState(TypedDict):
   """
   Represents the state of the graph, including the question, generation, web search, and retrieved documents.
   
   Attributes:
       question (str): The question being asked.
       generation (str): The generated response to the question.
       web_search (str): The results of any web search performed.
       documents (List[str]): A list of retrieved documents relevant to the question.
   """ 

   question: str
   generation: str
   web_search: bool
   documents: List[str]