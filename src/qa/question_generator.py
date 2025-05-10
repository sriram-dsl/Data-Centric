from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

def generate_questions(llm: OllamaLLM, retriever, query: str, num_questions: int) -> list[str]:
    """Generate analytical questions based on e-commerce data."""
    question_gen_template = """
    You are an expert in creating mathematical word problems like those in the GSM8K dataset.
    
    Based on the following e-commerce data context, create {num_questions} diverse word problems that:
    1. Require mathematical reasoning and calculations (arithmetic, percentages, rates)
    2. Are self-contained with all necessary information to solve
    3. Tell a brief story or scenario about e-commerce analytics
    4. Have a clear, single numerical answer
    5. Focus on business metrics and customer behavior
    
    CONTEXT INFORMATION:
    {context}
    
    INSTRUCTIONS:
    - Create word problems like those found in GSM8K dataset
    - Include specific numerical values needed to solve the problem
    - Avoid referencing external data or "according to data" phrases
    - Use realistic scenarios from e-commerce (sales, customer metrics, marketing results)
    - Questions should be clearly written and unambiguous
    - Focus on numbers, percentages, and business metrics
    
    EXAMPLE GSM8K-STYLE QUESTIONS:
    1. An online store sold 240 items in the electronics category and 180 items in the clothing category last month. If electronics items cost $85 on average and clothing items cost $45 on average, what was the total revenue from both categories?
    2. An e-commerce website has 850 total customers. If 42% of customers are in the loyalty program and loyalty program members spend $78 on average per order while non-members spend $52 on average, how much more revenue does the store generate from loyalty members compared to non-members if each customer makes exactly one order?
    
    FORMAT:
    1. Question 1
    2. Question 2
    (and so on)
    
    QUESTIONS:
    """
    
    question_gen_prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template=question_gen_template,
    )
    
    print(f"Retrieving context for query: '{query}'")
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    print(f"Generating {num_questions} questions...")
    response = llm.invoke(
        question_gen_prompt.format(
            context=context_text,
            num_questions=num_questions
        )
    )
    
    questions = []
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line and (line.startswith('Q') or line.startswith('Question') or (line[0].isdigit() and '.' in line[:3])):
            clean_question = line
            if line[0].isdigit() and '.' in line[:3]:
                clean_question = line.split('.', 1)[1].strip()
            elif line.startswith('Question'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    clean_question = parts[1].strip()
            questions.append(clean_question)
    
    if not questions:
        questions = [line.strip() for line in response.split('\n') if '?' in line]
    
    if not questions:
        print("Warning: Could not extract questions, using raw response")
        return [response.strip()]
    
    return questions[:num_questions]
