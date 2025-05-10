from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

def answer_question(llm: OllamaLLM, retriever, question: str) -> str:
    """Answer a question using the e-commerce RAG system with GSM8K-style reasoning."""
    answering_template = """
    You are an expert mathematician solving word problems in the style of GSM8K dataset answers.
    
    QUESTION:
    {question}
    
    Use the following e-commerce data to enhance your answer if needed:
    {context}
    
    INSTRUCTIONS:
    1. Use step-by-step reasoning to solve the problem
    2. Start each step with concise explanations of your thinking
    3. Show all calculations clearly with "X operation Y = Z" format
    4. Use precise arithmetic with no rounding until the final answer
    5. Your final answer should be just the number (with units if appropriate)
    
    EXAMPLE GSM8K-STYLE SOLUTION:
    Question: An online store sold 240 items in the electronics category and 180 items in the clothing category last month. If electronics items cost $85 on average and clothing items cost $45 on average, what was the total revenue from both categories?
    
    Answer:
    Electronics revenue = 240 * $85 = $20,400
    Clothing revenue = 180 * $45 = $8,100
    Total revenue = $20,400 + $8,100 = $28,500
    The total revenue from both categories is $28,500.
    
    YOUR STEP-BY-STEP SOLUTION:
    """
    
    answer_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=answering_template,
    )
    
    print(f"Retrieving context for question: '{question[:50]}...'")
    docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    print("Generating answer...")
    response = llm.invoke(
        answer_prompt.format(
            question=question,
            context=context_text
        )
    )
    
    return response
