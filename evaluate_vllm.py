import requests
import json
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

def query_vllm(vllm_url, model_name, prompt, max_tokens=50, temperature=0.7):
    """Sends a prompt to the VLLM server and returns the generated answer."""
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    })
    try:
        response = requests.post(vllm_url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()['choices']['text'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying VLLM: {e}")
        return None
    except KeyError:
        print("Error: Could not extract the answer from the VLLM response.")
        return None

def extract_questions_answers(dataset_name="anon8231489123/ShareGPT_Vicuna_unfiltered", num_samples=100):
    """Extracts a specified number of question-answer pairs from the ShareGPT dataset."""
    dataset = load_dataset(dataset_name)
    questions =
    reference_answers =
    count = 0
    for example in dataset['train']:
        if count >= num_samples:
            break
        if 'conversations' in example:
            conversation = example['conversations']
            if len(conversation) >= 2 and conversation['from'] == 'human' and conversation[1]['from'] == 'gpt':
                questions.append(conversation['value'])
                reference_answers.append(conversation[1]['value'])
                count += 1
        elif 'messages' in example:
            messages = example['messages']
            if len(messages) >= 2 and messages['role'] == 'user' and messages[1]['role'] == 'assistant':
                questions.append(messages['content'])
                reference_answers.append(messages[1]['content'])
                count += 1
    return questions, reference_answers

def calculate_bleu(reference, generated):
    """Calculates the BLEU score between a reference and a generated sentence."""
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothing_function)

def calculate_rouge_l(reference, generated):
    """Calculates the ROUGE-L score between a reference and a generated sentence."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores['rougeL'].fmeasure

def calculate_semantic_similarity(reference, generated, model_name='all-MiniLM-L6-v2'):
    """Calculates the semantic similarity between a reference and a generated sentence."""
    model = SentenceTransformer(model_name)
    reference_embedding = model.encode(reference)
    generated_embedding = model.encode(generated)
    similarity = cosine_similarity([reference_embedding], [generated_embedding])
    return similarity

def main():
    vllm_url = "http://localhost:8000/v1/completions"  # Default VLLM server URL
    model_name = "facebook/opt-125m"  # Replace with the model you are serving
    num_questions = 10  # Number of questions to evaluate

    questions, reference_answers = extract_questions_answers(num_samples=num_questions)

    if not questions:
        print("No questions found in the dataset.")
        return

    generated_answers =
    for question in questions:
        answer = query_vllm(vllm_url, model_name, question)
        if answer:
            generated_answers.append(answer)
        else:
            generated_answers.append("")

    bleu_scores =
    rouge_l_scores =
    semantic_similarities =

    print("\n--- Evaluation Results ---")
    for i in range(len(questions)):
        question = questions[i]
        reference = reference_answers[i]
        generated = generated_answers[i]

        bleu = calculate_bleu(reference, generated)
        rouge_l = calculate_rouge_l(reference, generated)
        semantic_similarity = calculate_semantic_similarity(reference, generated)

        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)
        semantic_similarities.append(semantic_similarity)

        print(f"\nQuestion: {question}")
        print(f"Reference Answer: {reference}")
        print(f"Generated Answer: {generated}")
        print(f"BLEU Score: {bleu:.4f}")
        print(f"ROUGE-L Score: {rouge_l:.4f}")
        print(f"Semantic Similarity: {semantic_similarity:.4f}")

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0

    print("\n--- Average Scores ---")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
    print(f"Average Semantic Similarity: {avg_semantic_similarity:.4f}")

if __name__ == "__main__":
    main()
