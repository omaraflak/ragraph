chunks_relationship = """I will give you two chunks of texts. Output another piece of text (maximum 100 words) to represent the semantic relationship between the two chuncks of text.
The generated text must be simple, objective, and should not provide any extra context that is not provided by the input texts.
Please take into consideration that sometimes the two chunks may not have any semantic or logic relationship; In that case, please anser that the two chunchs have no relationship, and avoid at all cost to create a imaginary/erronous connection or relationship.
However if you think the two chuncks have a distant relationship, feel free to briefly mention this relationship.

Example:
chunk 1: E=m*c^2 is the mass energy physic equation.
chunk 2: Albert Einstein wrote the mass energy physic equation.
relationship: Albert Einstein wrote the mass energy physic equation E=m*c^2.

Example:
chunk 1: The speed of light is 3x10^8 metre per second.
chunk 2: I love bananas.
relationship: There is no relationship. 

Example:
chunk 1: Barack Obama was the 44th president of the United States.
chunk 2: Donald Trump was the 45th president of the United States.
relationship: Barack Obama and Donald Trump were both United States presidents, Donald Trump was Barack Obama successor. 

Now it's your turn:
chunk 1: {chunk1}
chunk 2: {chunk2}
relationship: """

question_answer_generation = """Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Example:
context: The exponential function in math is written as exp(x) or e^x, and it has the fundamental property that it is equal to its rate of change at any point x.
question: What is the fundamental property of the exponential function in math?
answer: The fundamental property of the exponential function in math is that it is equal to its own rate of change at any point.

Now it's your turn:
context: {context}
"""

groundedness_scoring = """You will be given a context and a question.
Your task is to provide a 'rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.
You are also asked to give a rationale for the rating, as a text.

Example:
question: How is written the exponential function in math?
context: The exponential function, denoted as exp(x), is a mathematical function that has the property to be equal to it's rate of change.
evaluation: The context specifies that the exponential function is written as exp(x).
rating: 5

Example:
question: Who is the fifth president of the United States of America?
context: George Bush Junior was a terrible president.
evaluation: The context includes information about a president, but it is unknown whether he is a president of the Unites States of America, nor if he was the fifth.
rating: 1

Now it's your turn:
question: {question}
context: {context}
"""

relevance_scoring = """You will be given a question.
Your task is to provide a 'rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Example:
question: How is written the exponential function in math?
evaluation: The context specifies that the exponential function is written as exp(x).
rating: 5

Now it's your turn:
question: {question}
"""

standalone_scoring = """You will be given a question.
Your task is to provide a 'rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

Example:
question: What is the name of the checkpoint from which the ViT model is imported
evaluation: There is an implicit mention of a context, thus the question is not independant from the context
rating: 1

Example:
question: How is written the exponential function in math?
evaluation: The question is free of context.
rating: 5

Now it's your turn:
question: {question}
"""

question_with_context = """Using the information contained in the context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.

<Context>
{context}

<Question>
{question}

<Answer>
"""