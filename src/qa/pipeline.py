import os
import json
import time
from typing import List, Dict
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from src.qa.question_generator import generate_questions
from src.qa.answer_generator import answer_question
from src.qa.qa_formatter import format_qa_pair, validate_single_qa_pair

class EcommerceQAPairGenerator:
    """Automated pipeline for generating QA pairs from e-commerce data using RAG."""
    
    def __init__(self, vector_store_path: str, llm_model: str, output_dir: str, num_questions_per_category: int):
        self.vector_store_path = vector_store_path
        self.llm_model = llm_model
        self.output_dir = output_dir
        self.num_questions_per_category = num_questions_per_category
        os.makedirs(output_dir, exist_ok=True)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, embeddings, and vector store."""
        print("Initializing pipeline components...")
        self.llm = OllamaLLM(model=self.llm_model)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        try:
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            print(f"Successfully loaded vector store from {self.vector_store_path}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def run_pipeline(self, categories: List[str], num_questions_total: int) -> List[Dict[str, str]]:
        """Run the complete QA pair generation pipeline."""
        all_formatted_qa_pairs = []
        questions_per_category = min(self.num_questions_per_category, max(1, num_questions_total // len(categories)))
        
        print(f"Starting pipeline to generate {num_questions_total} total QA pairs")
        print(f"Will generate {questions_per_category} questions per category")
        
        try:
            for category in categories:
                if len(all_formatted_qa_pairs) >= num_questions_total:
                    break
                print(f"\n{'='*50}\nProcessing category: {category}\n{'='*50}")
                
                questions = generate_questions(self.llm, self.retriever, category, questions_per_category)
                print(f"Generated {len(questions)} questions for category: {category}")
                
                for i, question in enumerate(questions):
                    if len(all_formatted_qa_pairs) >= num_questions_total:
                        break
                    print(f"\n{'-'*50}\nProcessing question {i+1}/{len(questions)}: {question[:100]}...")
                    
                    try:
                        max_attempts = 2
                        for attempt in range(max_attempts):
                            try:
                                answer = answer_question(self.llm, self.retriever, question)
                                formatted_qa = format_qa_pair(self.llm, question, answer)
                                
                                if validate_single_qa_pair(formatted_qa):
                                    qa_pair = {
                                        "original_question": question,
                                        "original_answer": answer,
                                        "formatted_question": formatted_qa["question"],
                                        "formatted_answer": formatted_qa["answer"]
                                    }
                                    all_formatted_qa_pairs.append({
                                        "question": formatted_qa["question"],
                                        "answer": formatted_qa["answer"]
                                    })
                                    with open(f"{self.output_dir}/qa_pair_{len(all_formatted_qa_pairs)}.json", "w") as f:
                                        json.dump(qa_pair, f, indent=2)
                                    print(f"Successfully processed and saved QA pair {len(all_formatted_qa_pairs)}")
                                    break
                                else:
                                    print(f"Attempt {attempt+1}: QA pair failed validation, trying again")
                                    if attempt == max_attempts - 1:
                                        print(f"Skipping question after {max_attempts} failed attempts")
                            except Exception as e:
                                print(f"Error in attempt {attempt+1}: {e}")
                                if attempt == max_attempts - 1:
                                    print(f"Skipping question after {max_attempts} failed attempts")
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Error processing question: {e}")
                        continue
            
            gsm8k_format_pairs = self.convert_to_gsm8k_format(all_formatted_qa_pairs)
            final_output_path = f"{self.output_dir}/formatted_qa_pairs_final.json"
            with open(final_output_path, "w") as f:
                json.dump(all_formatted_qa_pairs, f, indent=2)
            gsm8k_output_path = f"{self.output_dir}/gsm8k_formatted_qa_pairs.json"
            with open(gsm8k_output_path, "w") as f:
                json.dump(gsm8k_format_pairs, f, indent=2)
            
            print(f"\nPipeline complete! Generated {len(all_formatted_qa_pairs)} QA pairs")
            print(f"Final output saved to: {final_output_path}")
            print(f"GSM8K format saved to: {gsm8k_output_path}")
            return gsm8k_format_pairs
        
        except Exception as e:
            print(f"Error in pipeline: {e}")
            if all_formatted_qa_pairs:
                recovery_path = f"{self.output_dir}/recovered_qa_pairs.json"
                with open(recovery_path, "w") as f:
                    json.dump(all_formatted_qa_pairs, f, indent=2)
                print(f"Saved {len(all_formatted_qa_pairs)} recovered QA pairs to: {recovery_path}")
            raise
    
    def convert_to_gsm8k_format(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert QA pairs to the exact format needed for GSM8K-style GPTO fine-tuning."""
        gsm8k_pairs = []
        for pair in qa_pairs:
            question = pair.get("question", "").strip()
            answer = pair.get("answer", "").strip()
            if "<<" not in answer or ">>" not in answer or "####" not in answer:
                print(f"Skipping pair with improper format: {question[:30]}...")
                continue
            gsm8k_pairs.append({"question": question, "answer": answer})
        print(f"Converted {len(gsm8k_pairs)}/{len(qa_pairs)} pairs to GSM8K format")
        return gsm8k_pairs
