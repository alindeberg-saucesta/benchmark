# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline inference throughput AND evaluate response quality."""
import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, List, Optional, Tuple, Union # Added List, Tuple

import torch
import uvloop
# --- Quality Metrics Imports ---
import numpy as np
from datasets import load_dataset as hf_load_dataset # Use alias to avoid conflict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# --- End Quality Metrics Imports ---

from benchmark_dataset import (AIMODataset, BurstGPTDataset,
                               ConversationDataset, InstructCoderDataset,
                               RandomDataset, SampleRequest, ShareGPTDataset,
                               SonnetDataset, VisionArenaDataset)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams # Added SamplingParams explicitly
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

# --- NLTK Data Download ---
# Ensure necessary NLTK data is available
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet')
    except (nltk.downloader.DownloadError, LookupError):
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except (nltk.downloader.DownloadError, LookupError):
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)

# --- Quality Evaluation Metrics ---
def calculate_bleu(reference: str, generated: str) -> float:
    """Calculates the BLEU score between a reference and a generated sentence."""
    if not reference or not generated: return 0.0
    try:
        reference_tokens = [nltk.word_tokenize(reference)]
        generated_tokens = nltk.word_tokenize(generated)
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_function)
    except Exception as e:
        # print(f"Error calculating BLEU: {e}") # Uncomment for debugging
        return 0.0

def calculate_rouge_l(reference: str, generated: str) -> float:
    """Calculates the ROUGE-L F-measure score."""
    if not reference or not generated: return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    except Exception as e:
        # print(f"Error calculating ROUGE-L: {e}") # Uncomment for debugging
        return 0.0

def calculate_semantic_similarity(reference: str, generated: str, model: SentenceTransformer) -> float:
    """Calculates the cosine similarity between sentence embeddings."""
    if not reference or not generated or model is None: return 0.0
    try:
        ref_emb = model.encode(reference)
        gen_emb = model.encode(generated)
        similarity = cosine_similarity([ref_emb], [gen_emb])
        return similarity[0][0]
    except Exception as e:
        # print(f"Error calculating Semantic Similarity: {e}") # Uncomment for debugging
        return 0.0

# --- vLLM Backend Functions (Modified slightly if needed) ---

def run_vllm(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    use_beam_search: bool = False, # Added beam search flag
    disable_detokenize: bool = False,
    # --- Quality Args ---
    temperature: float = 1.0, # Allow overriding temperature
    top_p: float = 1.0,       # Allow overriding top_p
    ignore_eos: bool = True, # Allow overriding ignore_eos
) -> tuple[float, Optional[list[RequestOutput]]]:
    from vllm import LLM # Keep import local

    # --- Pass EngineArgs directly ---
    llm = LLM( **dataclasses.asdict(engine_args))

    # Basic validation (can be enhanced)
    max_model_len = llm.llm_engine.model_config.max_model_len
    for req in requests:
         # If output_len is not explicitly set (e.g., for quality benchmark), use a default/max
        req_output_len = req.expected_output_len if req.expected_output_len is not None else (max_model_len - req.prompt_len - 5)
        if max_model_len < (req.prompt_len + req_output_len):
             warnings.warn(f"Request prompt ({req.prompt_len}) + output ({req_output_len}) exceeds max_model_len ({max_model_len}). Output might be truncated.")


    prompts: list[Union[TextPrompt, TokensPrompt]] = []
    sampling_params_list: list[Union[SamplingParams, BeamSearchParams]] = [] # Use Union type
    for request in requests:
        # Handle prompt format
        if "prompt_token_ids" in request.prompt and request.prompt["prompt_token_ids"] is not None:
             prompt_data = TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"],
                                        multi_modal_data=request.multi_modal_data)
        elif isinstance(request.prompt, str):
             prompt_data = TextPrompt(prompt=request.prompt,
                                      multi_modal_data=request.multi_modal_data)
        else:
            # Handle cases where request.prompt might be a dict but not with 'prompt_token_ids'
            # This might occur if quality benchmarking adapts SampleRequest differently.
            # Defaulting to string conversion, adjust as needed.
            prompt_text = str(request.prompt)
            prompt_data = TextPrompt(prompt=prompt_text,
                                     multi_modal_data=request.multi_modal_data)

        prompts.append(prompt_data)

        # Determine output length: Use expected_output_len if available for throughput,
        # otherwise use a potentially larger value for quality eval (or rely on SamplingParams default)
        # For quality, we often want the model to decide when to stop, so max_tokens can be large.
        # The original script *required* output_len. We make it optional here for quality.
        max_tokens_for_req = request.expected_output_len
        if max_tokens_for_req is None:
            # If quality benchmarking is enabled, use a larger default, otherwise error?
            # Let's use a reasonable default but allow SamplingParams default if needed
             max_tokens_for_req = 1024 # Default for quality, adjust as needed

        # Create SamplingParams or BeamSearchParams
        if use_beam_search:
            sampling_params_list.append(
                BeamSearchParams(
                    beam_width=n,
                    temperature=temperature, # Pass temperature
                    top_p=top_p, # Pass top_p
                    ignore_eos=ignore_eos,
                    max_tokens=max_tokens_for_req,
                    detokenize=not disable_detokenize,
                ))
        else:
            sampling_params_list.append(
                SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    ignore_eos=ignore_eos,
                    max_tokens=max_tokens_for_req,
                    detokenize=not disable_detokenize,
                    # stop=["<|eot_id|>", "\n"] # Add stop tokens if needed for quality
                ))

    lora_requests: Optional[list[LoRARequest]] = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]


    outputs: Optional[list[RequestOutput]] = None
    start = time.perf_counter()

    # Use generate for both SamplingParams and BeamSearchParams
    outputs = llm.generate(prompts,
                           sampling_params_list,
                           lora_request=lora_requests,
                           use_tqdm=True)

    end = time.perf_counter()
    return end - start, outputs


# run_vllm_chat function can be similarly adapted if needed, passing temperature/top_p etc.
# For simplicity, we'll focus quality benchmarking on the standard `run_vllm` for now.
async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    # --- Quality Args ---
    temperature: float = 1.0,
    top_p: float = 1.0,
    ignore_eos: bool = True,
    # --- End Quality Args ---
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
) -> float:
    # NOTE: Async version doesn't easily return RequestOutput needed for quality eval.
    # Quality benchmarking might be harder/impossible here without modifications
    # to how results are collected. Returning only elapsed time.
    print("Warning: Quality benchmarking is not directly supported with --async-engine.")

    from vllm import SamplingParams # Keep import local

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:

        # --- Validation (same as sync) ---
        max_model_len = llm.model_config.max_model_len
        for req in requests:
            req_output_len = req.expected_output_len if req.expected_output_len is not None else (max_model_len - req.prompt_len - 5)
            if max_model_len < (req.prompt_len + req_output_len):
                 warnings.warn(f"Request prompt ({req.prompt_len}) + output ({req_output_len}) exceeds max_model_len ({max_model_len}). Output might be truncated.")


        prompts: list[Union[TextPrompt, TokensPrompt]] = []
        sampling_params_list: list[SamplingParams] = []
        lora_requests: list[Optional[LoRARequest]] = []
        for request in requests:
            # Handle prompt format (same as sync)
            if "prompt_token_ids" in request.prompt and request.prompt["prompt_token_ids"] is not None:
                 prompt_data = TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"],
                                            multi_modal_data=request.multi_modal_data)
            elif isinstance(request.prompt, str):
                 prompt_data = TextPrompt(prompt=request.prompt,
                                          multi_modal_data=request.multi_modal_data)
            else:
                prompt_text = str(request.prompt)
                prompt_data = TextPrompt(prompt=prompt_text,
                                         multi_modal_data=request.multi_modal_data)
            prompts.append(prompt_data)

            max_tokens_for_req = request.expected_output_len or 1024 # Default

            # Create SamplingParams
            sampling_params_list.append(
                SamplingParams(
                    n=n,
                    temperature=temperature, # Use provided temp
                    top_p=top_p,           # Use provided top_p
                    ignore_eos=ignore_eos,       # Use provided ignore_eos
                    max_tokens=max_tokens_for_req,
                    detokenize=not disable_detokenize,
                    # stop=["<|eot_id|>", "\n"] # Add stop tokens if needed
                ))
            lora_requests.append(request.lora_request)

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp,
                lr) in enumerate(zip(prompts, sampling_params_list, lora_requests)):
            generator = llm.generate(prompt,
                                     sp,
                                     lora_request=lr,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass # Consume results
        end = time.perf_counter()
        return end - start

# --- Dataset Loading (Modified) ---

def get_requests_and_references(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase
) -> Tuple[List[SampleRequest], Optional[List[str]]]:
    """
    Generates SampleRequests and optionally extracts reference answers
    if quality benchmarking is enabled and the dataset supports it.
    """
    requests: List[SampleRequest] = []
    reference_answers: Optional[List[str]] = None

    # --- Logic for Quality Benchmarking Datasets (e.g., ShareGPT) ---
    if args.run_quality_benchmark and args.dataset_name == "sharegpt":
        print(f"Loading dataset '{args.dataset_path}' for quality benchmarking...")
        try:
            # Load raw dataset using Hugging Face datasets
            # Adapt split/subset as needed
            raw_dataset = hf_load_dataset(args.dataset_path, split='train')
        except Exception as e:
            print(f"Error loading dataset '{args.dataset_path}' with Hugging Face datasets: {e}")
            print("Falling back to original benchmark_dataset loading (quality metrics will fail).")
            # Fallback to original method if HF loading fails or for other datasets
            requests = get_requests(args, tokenizer) # Call original function
            return requests, None

        questions = []
        references = []
        count = 0
        processed_indices = set() # Keep track of processed entries

        # Extract question-answer pairs (similar to the quality script)
        for i, example in enumerate(raw_dataset):
            if args.num_prompts is not None and count >= args.num_prompts:
                break

            question = None
            answer = None
            if 'conversations' in example and isinstance(example['conversations'], list) and len(example['conversations']) >= 2:
                conv = example['conversations']
                if (isinstance(conv[0], dict) and 'from' in conv[0] and conv[0]['from'] == 'human' and 'value' in conv[0] and
                    isinstance(conv[1], dict) and 'from' in conv[1] and conv[1]['from'] == 'gpt' and 'value' in conv[1]):
                    question = conv[0]['value']
                    answer = conv[1]['value']
            # Add other elif conditions here if needed for different dataset structures

            if question and answer:
                 # Basic filtering (optional, adapt as needed)
                 if len(question) > 10 and len(answer) > 10:
                     questions.append(question.strip())
                     references.append(answer.strip())
                     processed_indices.add(i)
                     count += 1

        print(f"Extracted {len(questions)} valid question/reference pairs for quality evaluation.")
        if not questions:
             print("Warning: No reference answers extracted. Quality benchmarking will be skipped.")
             # Fallback to standard request generation if no pairs found
             requests = get_requests(args, tokenizer)
             return requests, None

        reference_answers = references

        # Now, create SampleRequest objects using the extracted questions
        print("Tokenizing prompts for SampleRequests...")
        for i, q_text in enumerate(tqdm(questions)):
            prompt_token_ids = tokenizer.encode(q_text)
            prompt_len = len(prompt_token_ids)
             # Quality benchmark often doesn't pre-define output length
            output_len = args.output_len # Use arg if provided, else None

            requests.append(SampleRequest(
                prompt=q_text,
                prompt_len=prompt_len,
                expected_output_len=output_len,
                prompt_token_ids=prompt_token_ids, # Store token IDs
                reference_answer=references[i] # Store reference (optional, maybe not needed in SampleRequest itself)
            ))
        print(f"Created {len(requests)} SampleRequests for quality benchmarking.")

    # --- Fallback or Default Logic (Use original get_requests) ---
    else:
        if args.run_quality_benchmark:
            print(f"Warning: Quality benchmarking enabled but not implemented for dataset '{args.dataset_name}'.")
            print("Throughput benchmark will run, but quality scores will not be calculated.")
        # Use the original get_requests function from benchmark_dataset.py (or defined below)
        requests = get_requests(args, tokenizer)
        # No reference answers available in this path
        reference_answers = None


    # --- Ensure num_prompts limit is respected if applied post-extraction ---
    if args.num_prompts is not None and len(requests) > args.num_prompts:
         print(f"Limiting requests from {len(requests)} to {args.num_prompts}.")
         requests = requests[:args.num_prompts]
         if reference_answers is not None:
             reference_answers = reference_answers[:args.num_prompts]

    return requests, reference_answers

# --- Keep the original get_requests for non-quality scenarios ---
# (Copied from the original script provided by the user for clarity)
def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if args.backend == "vllm-chat":
            sample_kwargs["enable_multimodal_chat"] = True
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset.")
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs['dataset_split'] = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs['dataset_subset'] = args.hf_subset
            common_kwargs['dataset_split'] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
        else:
             raise ValueError(f"{args.dataset_path} is not supported by hf dataset.") # Added for robustness
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}

    # Instantiate and sample
    try:
        dataset = dataset_cls(**common_kwargs)
        return dataset.sample(**sample_kwargs)
    except Exception as e:
        print(f"Error initializing or sampling from {dataset_cls.__name__}: {e}")
        # Depending on desired behavior, either raise e or return an empty list
        return []


# --- Main Function (Modified) ---
def main(args: argparse.Namespace):
    if args.seed is None:
        args.seed = 0
    print("--- Benchmark Configuration ---")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed) # Seed numpy as well for consistency if needed

    # --- Setup for Quality Benchmarking ---
    semantic_model = None
    if args.run_quality_benchmark:
        print("\n--- Quality Benchmarking Enabled ---")
        download_nltk_data() # Ensure NLTK data is present
        if args.semantic_model_name:
            print(f"Loading semantic similarity model: {args.semantic_model_name}...")
            try:
                semantic_model = SentenceTransformer(args.semantic_model_name)
                print("Semantic model loaded.")
            except Exception as e:
                print(f"Error loading semantic model '{args.semantic_model_name}': {e}")
                print("Semantic similarity calculation will be skipped.")
                semantic_model = None
        else:
            print("Warning: --semantic-model-name not provided. Semantic similarity will be skipped.")

    # --- Load Tokenizer and Data ---
    print("\n--- Loading Tokenizer and Data ---")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    # Use the modified function to potentially get references
    requests, reference_answers = get_requests_and_references(args, tokenizer)

    if not requests:
        print("Error: No requests generated. Exiting.")
        return

    # Check if quality benchmarking is feasible
    can_run_quality = (args.run_quality_benchmark and
                      reference_answers is not None and
                      len(requests) == len(reference_answers) and
                      args.backend in {"vllm", "vllm-chat"}) # Only support vllm sync backends for now

    if args.run_quality_benchmark and not can_run_quality:
        print("\nWarning: Cannot run quality benchmark.")
        if reference_answers is None: print("- Reference answers not available for the selected dataset/configuration.")
        if len(requests) != len(reference_answers or []): print("- Mismatch between number of requests and references.")
        if args.backend not in {"vllm", "vllm-chat"}: print(f"- Backend '{args.backend}' not currently supported for quality eval output extraction.")
        print("Proceeding with throughput benchmark only.")


    is_multi_modal = any(getattr(request, 'multi_modal_data', None) is not None
                         for request in requests)
    request_outputs: Optional[list[RequestOutput]] = None
    elapsed_time: float = 0.0

    print(f"\n--- Running Backend: {args.backend} ---")
    # --- Backend Execution ---
    if args.backend == "vllm":
        # Pass quality-related sampling params if benchmarking quality
        temp = args.temperature if can_run_quality else 1.0
        tp = args.top_p if can_run_quality else 1.0
        ignore_eos_flag = args.ignore_eos if can_run_quality else True # Be careful with ignore_eos for quality

        if args.async_engine:
            elapsed_time = uvloop.run(
                run_vllm_async(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    temperature=temp,
                    top_p=tp,
                    ignore_eos=ignore_eos_flag,
                    disable_frontend_multiprocessing=args.disable_frontend_multiprocessing,
                    disable_detokenize=args.disable_detokenize,
                ))
        else:
            elapsed_time, request_outputs = run_vllm(
                requests, args.n, EngineArgs.from_cli_args(args),
                use_beam_search=(args.n > 1 and args.use_beam_search), # Pass beam search flag
                temperature=temp,
                top_p=tp,
                ignore_eos=ignore_eos_flag,
                disable_detokenize=args.disable_detokenize)

    # ... (elif blocks for hf, mii, vllm-chat - keep original logic)
    elif args.backend == "hf":
         # Quality eval not implemented for HF backend here
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.hf_max_batch_size, args.trust_remote_code,
                              args.disable_detokenize)
    elif args.backend == "mii":
         # Quality eval not implemented for MII backend here
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len) # Note: MII might need fixed output len
    elif args.backend == "vllm-chat":
         # Quality eval *could* be added here similarly to run_vllm
         # Pass quality-related sampling params if benchmarking quality
        temp = args.temperature if can_run_quality else 1.0
        tp = args.top_p if can_run_quality else 1.0
        ignore_eos_flag = args.ignore_eos if can_run_quality else True

        # We need run_vllm_chat to accept these params, assuming it's adapted
        # elapsed_time, request_outputs = run_vllm_chat(
        #      requests, args.n, EngineArgs.from_cli_args(args),
        #      temperature=temp, top_p=tp, ignore_eos=ignore_eos_flag,
        #      disable_detokenize=args.disable_detokenize
        # )
        # Using original call signature for now as run_vllm_chat wasn't provided for modification
        print("Warning: Assuming run_vllm_chat uses default sampling for quality eval.")
        elapsed_time, request_outputs = run_vllm_chat(
             requests, args.n, EngineArgs.from_cli_args(args),
             args.disable_detokenize)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


    print(f"\n--- Throughput Results ---")
    # --- Throughput Calculation (Original Logic) ---
    if request_outputs and args.backend in {"vllm", "vllm-chat"}:
        total_prompt_tokens = 0
        total_output_tokens = 0
        for ro in request_outputs:
            if not isinstance(ro, RequestOutput): continue
            total_prompt_tokens += len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
            total_output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)
        total_num_tokens = total_prompt_tokens + total_output_tokens
        # Note: If ignore_eos=False for quality, output tokens might be less than expected_output_len
    else:
        # Fallback estimation if no detailed outputs (e.g., async, hf, mii)
        total_num_tokens = sum((r.prompt_len or 0) + (r.expected_output_len or 0) for r in requests)
        total_output_tokens = sum(r.expected_output_len or 0 for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens
        print("Warning: Token counts are estimated based on input lengths for this backend.")


    if is_multi_modal and args.backend != "vllm-chat":
        print("\033[91mWARNING\033[0m: Multi-modal request with "
              f"{args.backend} backend detected. Throughput metrics might be "
              "inaccurate if image tokens aren't counted correctly by the backend.")

    req_per_sec = len(requests) / elapsed_time
    total_tok_per_sec = total_num_tokens / elapsed_time
    output_tok_per_sec = total_output_tokens / elapsed_time

    print(f"Elapsed time: {elapsed_time:.2f} s")
    print(f"Total prompts processed: {len(requests)}")
    print(f"Throughput: {req_per_sec:.2f} requests/s, "
          f"{total_tok_per_sec:.2f} total tokens/s, "
          f"{output_tok_per_sec:.2f} output tokens/s")
    print(f"Total num prompt tokens: {total_prompt_tokens}")
    print(f"Total num output tokens: {total_output_tokens}")

    # --- Quality Score Calculation ---
    quality_results = {}
    if can_run_quality and request_outputs:
        print("\n--- Quality Evaluation Results ---")
        bleu_scores = []
        rouge_l_scores = []
        semantic_similarities = []
        generated_texts_for_eval = [] # Store generated texts

        # Check alignment before proceeding
        if len(request_outputs) != len(reference_answers):
             print(f"Error: Mismatch between request outputs ({len(request_outputs)}) and references ({len(reference_answers)}). Skipping quality eval.")
        else:
            print("Calculating scores...")
            for i, ro in enumerate(tqdm(request_outputs)):
                 if not isinstance(ro, RequestOutput) or not ro.outputs:
                     # Handle cases where a request might have failed or produced no output
                     generated = ""
                 else:
                     # Assuming n=1 or we take the first output
                     generated = ro.outputs[0].text.strip()

                 reference = reference_answers[i]
                 generated_texts_for_eval.append(generated) # Keep for potential later inspection

                 bleu = calculate_bleu(reference, generated)
                 rouge_l = calculate_rouge_l(reference, generated)
                 sem_sim = calculate_semantic_similarity(reference, generated, semantic_model)

                 bleu_scores.append(bleu)
                 rouge_l_scores.append(rouge_l)
                 semantic_similarities.append(sem_sim)

                 # Optional: Print individual scores (can be very verbose)
                 # print(f"\n--- Sample {i+1} ---")
                 # print(f"Reference: {reference[:100]}...")
                 # print(f"Generated: {generated[:100]}...")
                 # print(f"BLEU: {bleu:.4f}, ROUGE-L: {rouge_l:.4f}, SemSim: {sem_sim:.4f}")


            # Calculate Averages
            avg_bleu = np.nanmean(bleu_scores) if bleu_scores else 0
            avg_rouge_l = np.nanmean(rouge_l_scores) if rouge_l_scores else 0
            avg_semantic_similarity = np.nanmean(semantic_similarities) if semantic_similarities else 0

            print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
            print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
            print(f"Average Semantic Similarity: {avg_semantic_similarity:.4f}")

            quality_results = {
                "avg_bleu": avg_bleu,
                "avg_rouge_l": avg_rouge_l,
                "avg_semantic_similarity": avg_semantic_similarity,
                # Optionally add lists of individual scores if needed in JSON
                # "bleu_scores": bleu_scores,
                # "rouge_l_scores": rouge_l_scores,
                # "semantic_similarities": semantic_similarities,
            }
            # Optional: Save generated texts if needed
            # with open("generated_texts.json", "w") as f:
            #     json.dump({"references": reference_answers, "generated": generated_texts_for_eval}, f, indent=2)


    # --- Output JSON Results ---
    if args.output_json:
        print(f"\nSaving results to {args.output_json}...")
        results = {
            "config": vars(args), # Store arguments used
            "throughput": {
                "elapsed_time_s": elapsed_time,
                "num_requests": len(requests),
                "total_num_tokens": total_num_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_output_tokens": total_output_tokens,
                "requests_per_second": req_per_sec,
                "total_tokens_per_second": total_tok_per_sec,
                "output_tokens_per_second": output_tok_per_sec,
            },
             # Add quality results if they were calculated
            "quality": quality_results if quality_results else None,
        }
        # Clean up non-serializable args if any (e.g., functions, complex objects)
        if "config" in results:
             results["config"] = {k: v for k, v in results["config"].items()
                                  if isinstance(v, (str, int, float, bool, list, dict, type(None)))}

        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

        # Save PyTorch benchmark format (optional, focuses on throughput)
        # save_to_pytorch_benchmark_format(args, results["throughput"]) # Pass throughput dict

    print("\n--- Benchmark Complete ---")


# --- Argument Parsing (Modified) ---
if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark throughput and optionally evaluate quality.")

    # --- Original Arguments ---
    parser.add_argument("--backend", type=str, choices=["vllm", "hf", "mii", "vllm-chat"], default="vllm")
    parser.add_argument("--dataset-name", type=str, choices=["sharegpt", "random", "sonnet", "burstgpt", "hf"], default="sharegpt", help="Name of the dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset (DEPRECATED, use --dataset-path).")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path or Hugging Face identifier for the dataset.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length (required for 'random' dataset).")
    parser.add_argument("--output-len", type=int, default=None, help="Output length. If None, determined by dataset or quality defaults.")
    parser.add_argument("--n", type=int, default=1, help="Number of generated sequences per prompt (for beam search or sampling).")
    parser.add_argument("--use-beam-search", action="store_true", help="Use beam search instead of sampling (requires n > 1).")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size", type=int, default=None, help="Maximum batch size for HF backend.")
    parser.add_argument('--output-json', type=str, default=None, help='Path to save results in JSON format.')
    parser.add_argument("--async-engine", action='store_true', help="Use vLLM async engine (quality eval disabled).")
    parser.add_argument("--disable-frontend-multiprocessing", action='store_true', help="Disable decoupled async engine frontend.")
    parser.add_argument("--disable-detokenize", action="store_true", help="Do not detokenize response.")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapters.")
    parser.add_argument("--prefix-len", type=int, default=None, help="Prefix tokens for Random/Sonnet datasets.")
    parser.add_argument("--random-range-ratio", type=float, default=None, help="Range ratio for RandomDataset.")
    parser.add_argument("--hf-subset", type=str, default=None, help="Subset of the HF dataset.")
    parser.add_argument("--hf-split", type=str, default=None, help="Split of the HF dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    # --- Add VLLM Engine Arguments ---
    parser = AsyncEngineArgs.add_cli_args(parser) # Use AsyncEngineArgs as it includes EngineArgs

    # --- Quality Benchmarking Arguments ---
    parser.add_argument("--run-quality-benchmark", action="store_true", help="Enable quality benchmarking (BLEU, ROUGE, SemSim). Requires suitable dataset (e.g., sharegpt) and sync vLLM backend.")
    parser.add_argument("--semantic-model-name", type=str, default='all-MiniLM-L6-v2', help="Sentence Transformer model for semantic similarity.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (used if quality benchmark enabled).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p (used if quality benchmark enabled).")
    parser.add_argument("--ignore-eos", action="store_true", default=False, help="Ignore EOS token during generation (used if quality benchmark enabled). Set carefully.")


    args = parser.parse_args()

    # --- Argument Validation and Defaults ---
    if args.tokenizer is None:
        args.tokenizer = args.model
    # validate_args(args) # Original validation can be added back if needed

    # Basic validation for quality benchmark
    if args.run_quality_benchmark:
        if args.backend not in {"vllm", "vllm-chat"}:
             print(f"Warning: Quality benchmarking requested but backend '{args.backend}' might not provide necessary outputs. Trying anyway...")
        if args.async_engine:
             print("Warning: --run-quality-benchmark is not compatible with --async-engine. Disabling quality benchmark.")
             args.run_quality_benchmark = False
        if args.dataset_name != "sharegpt":
             # Add more supported datasets here if logic is implemented in get_requests_and_references
             print(f"Warning: Quality benchmarking reference extraction currently only implemented for --dataset-name sharegpt. Attempting standard run for '{args.dataset_name}'.")
             # We allow it to proceed, but get_requests_and_references will likely return None for references

    # If output_len is not set and quality benchmark is off, it might cause issues
    # with the original dataset sampling logic which expected it for some datasets (like random).
    if args.output_len is None and not args.run_quality_benchmark and args.dataset_name == "random":
         warnings.warn("Running random dataset without --output-len and without quality benchmarking. Output length might be unpredictable.")
         # Consider setting a default args.output_len = 1024 here if needed


    main(args)
