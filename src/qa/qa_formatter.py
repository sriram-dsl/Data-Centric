import re
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

def format_qa_pair(llm: OllamaLLM, question: str, answer: str) -> dict[str, str]:
    """Format a question-answer pair into the GSM8K-style format."""
    format_template = """
    You are an expert in formatting mathematical problems and solutions to match the GSM8K dataset format for GPTO fine-tuning.
    
    Transform this e-commerce analytics question and answer to match the GSM8K format exactly.
    
    ORIGINAL QUESTION:
    {question}
    
    ORIGINAL ANSWER:
    {answer}
    
    GSM8K FORMAT REQUIREMENTS:
    
    1. The QUESTION must:
       - Be a self-contained word problem with all needed values
       - Read like a real-world scenario without referencing external data
       - Have clear numerical values that can be used in calculations
       - End with a clear mathematical question
    
    2. The ANSWER must follow this EXACT format:
       - Multiple steps of reasoning, each on its own line
       - Each calculation should be written in this format: "X operation Y = result"
       - Every calculation that's shown must be embedded in "<<calculation=result>>" format
       - For example: "Total customers = 240 + 180 = <<240+180=420>>420"
       - The final line MUST be "#### [numerical answer]" with just the number
       
    EXAMPLE GSM8K-FORMATTED QUESTION AND ANSWER:
    
    question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    
    answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72
    
    ANOTHER EXAMPLE:
    
    question: An online store had 240 female customers who used discount codes. If this represents 53.1% of all female customers, how many female customers did not use discount codes?
    
    answer: First, I'll calculate the total number of female customers.
    Total female customers = 240 / 0.531 = <<240/0.531=451.98>>451.98 ≈ 452 customers
    
    Next, I'll find how many didn't use discounts.
    Female customers without discounts = 452 - 240 = <<452-240=212>>212 customers
    #### 212
    
    YOUR FORMATTED QA PAIR:
    question: [formatted question]
    
    answer: [step-by-step solution with <<calculation=result>> format for EVERY calculation]
    """
    
    format_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=format_template,
    )
    
    print("Formatting QA pair...")
    response = llm.invoke(
        format_prompt.format(
            question=question,
            answer=answer
        )
    )
    
    formatted_qa = {"question": "", "answer": ""}
    lines = response.lower().split('\n')
    question_index = -1
    answer_index = -1
    
    for i, line in enumerate(lines):
        if "question:" in line:
            question_index = i
        if "answer:" in line and i > question_index:
            answer_index = i
            break
    
    if question_index != -1 and answer_index != -1:
        question_text = lines[question_index].split("question:", 1)[1].strip()
        if question_index + 1 < answer_index:
            for j in range(question_index + 1, answer_index):
                question_text += " " + lines[j].strip()
        
        answer_lines = []
        answer_start = lines[answer_index].split("answer:", 1)[1].strip()
        if answer_start:
            answer_lines.append(answer_start)
        for j in range(answer_index + 1, len(lines)):
            answer_lines.append(lines[j].strip())
        
        formatted_qa["question"] = question_text
        formatted_qa["answer"] = "\n".join(answer_lines)
    else:
        print("Warning: Could not parse formatted QA pair, using fallback method")
        parts = response.split("question:", 1)
        if len(parts) > 1:
            rest = parts[1].strip()
            qa_parts = rest.split("answer:", 1)
            if len(qa_parts) > 1:
                formatted_qa["question"] = qa_parts[0].strip()
                formatted_qa["answer"] = qa_parts[1].strip()
    
    if not formatted_qa["question"] or not formatted_qa["answer"]:
        print("Warning: Formatted QA pair is incomplete, using original")
        formatted_qa["question"] = question
        formatted_qa["answer"] = "The answer is #### 100"
    
    if "<<" not in formatted_qa["answer"] or ">>" not in formatted_qa["answer"]:
        print("Warning: Answer lacks calculation format, fixing formatting")
        calculations = re.findall(r'(\d+\.?\d*)\s*[+\-*/]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', answer)
        if calculations:
            fixed_answer_lines = []
            for op1, op2, result in calculations:
                operation = "+" if "+" in answer[answer.find(op1):answer.find(result)] else \
                           "-" if "-" in answer[answer.find(op1):answer.find(result)] else \
                           "*" if "*" in answer[answer.find(op1):answer.find(result)] else "/"
                step_description = "Calculation"
                if "total" in answer.lower():
                    step_description = "Total"
                elif "average" in answer.lower():
                    step_description = "Average"
                elif "percentage" in answer.lower():
                    step_description = "Percentage"
                fixed_answer_lines.append(f"{step_description} = {op1} {operation} {op2} = <<{op1}{operation}{op2}={result}>>{result}")
            final_result = calculations[-1][2] if calculations else "100"
            fixed_answer_lines.append(f"#### {final_result}")
            formatted_qa["answer"] = "\n".join(fixed_answer_lines)
    
    if "####" not in formatted_qa["answer"]:
        print("Warning: Answer lacks final answer format, adding it")
        final_numbers = re.findall(r'(\d+\.?\d*)', formatted_qa["answer"])
        final_result = final_numbers[-1] if final_numbers else "100"
        formatted_qa["answer"] += f"\n#### {final_result}"
    
    return formatted_qa

def validate_single_qa_pair(qa_pair: dict[str, str]) -> bool:
    """Validate a single QA pair to ensure it meets GSM8K format."""
    question = qa_pair.get("question", "")
    answer = qa_pair.get("answer", "")
    
    has_question_mark = "?" in question
    has_calculation_format = "<<" in answer and ">>" in answer
    has_final_answer = "####" in answer
    has_numbers_in_question = any(char.isdigit() for char in question)
    has_data_reference = "according to the data" in question.lower() or "the data shows" in question.lower()
    has_segment_reference = "segment analysis" in question.lower() and not any(char.isdigit() for char in question)
    has_placeholder_text = "[insert" in answer or "unknown" in answer.lower() or "no calculation needed" in answer
    final_answer_pattern = r'####\s*\$?(\d+\.?\d*%?)'
    has_proper_final_answer = bool(re.search(final_answer_pattern, answer))
    has_sufficient_length = len(question) >= 80
    
    is_valid = (
        has_question_mark and 
        has_calculation_format and 
        has_final_answer and 
        has_numbers_in_question and 
        not has_data_reference and 
        not has_segment_reference and 
        not has_placeholder_text and
        has_proper_final_answer and
        has_sufficient_length
    )
    
    if not is_valid:
        print("Validation failures:")
        if not has_question_mark: print("- Missing question mark")
        if not has_calculation_format: print("- Missing <<calculation=result>> format")
        if not has_final_answer: print("- Missing #### format")
        if not has_proper_final_answer: print("- Final answer not in proper #### format")
        if not has_numbers_in_question: print("- No numbers in question")
        if has_data_reference: print("- References external data")
        if has_segment_reference: print("- References segment analysis without numbers")
        if has_placeholder_text: print("- Contains placeholder text")
        if not has_sufficient_length: print("- Question too short, likely missing context")
    
    return is_valid
