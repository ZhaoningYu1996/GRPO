import gc
import logging
import math
import json
from array import array
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import fire
import torch
from huggingface_hub import repo_exists
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VllmInference:
    def __init__(
        self,
        model_name: str,
        distributed_executor_backend="ray",
        tokenizer: Optional[str] = None,
        tensor_parallel_size: int = 4,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: int = None,
        max_tokens: int = 2048,
    ) -> None:


        self.vllm_model = self.init_model(
            model=model_name,
            distributed_executor_backend=distributed_executor_backend,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
        )


        self.sampling_parameters = SamplingParams(
            n=1,  # THIS MUST STAY n = 1
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )


        logger.info(f"Loaded {model_name} with {self.sampling_parameters}")


    def init_model(
        self,
        model: str,
        distributed_executor_backend: str,
        tokenizer: Optional[str],
        tensor_parallel_size: int,
    ):
        if not Path(model).exists():
            try:
                repo_exists_result = repo_exists(model)
            except:
                repo_exists_result = False
            if not repo_exists_result:
                raise ValueError(
                    f"Could not find model {model} of HF or locally. Locally supported models are: {list(MODEL_NAMES_TO_PATHS.keys())}"
                )
        llm_args = {
            "model": model,
            "distributed_executor_backend": distributed_executor_backend,
            "tokenizer": tokenizer,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": True,
        }
        llm_instance = LLM(**llm_args)
        return llm_instance


    def reset_sampling_parameters(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int = None,
        max_tokens: int = 32768,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        # reset sampling parameters on the fly without recreating entire class
        self.sampling_parameters = SamplingParams(
            n=1,  # THIS MUST STAY n = 1
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )


    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
    ):
        return self.vllm_model.generate(
            prompts=prompts,
            prompt_token_ids=prompt_token_ids,
            sampling_params=self.sampling_parameters,
        )




def extract_vllm_output(request_output):
    token_ids = request_output.outputs[0].token_ids
    result = {
        "output": request_output.outputs[0].text,
        "token_ids": token_ids.tolist() if isinstance(token_ids, array) else token_ids,
        "cumulative_logprob": request_output.outputs[0].cumulative_logprob,
        "finish_reason": request_output.outputs[0].finish_reason,
        "prompt_token_ids": getattr(request_output, "prompt_token_ids", None),
    }
    return result

def update_serialized_vllm_output(ex, vllm_output):
    ex["vllm_output"] = extract_vllm_output(vllm_output)
    return ex

def format_prompt(example, tokenizer):
    curr_prompt = tokenizer.apply_chat_template(
       example['prompt'], tokenize=False, add_generation_prompt=True, enable_thinking=True
   )
    # if tokenizer.bos_token is not None:
    #    curr_prompt = curr_prompt.replace(tokenizer.bos_token, "")
    return {
        "input": curr_prompt,
        "solution": example["reward_model"]["ground_truth"],
    }


def vllm_generate(
    prompt_path: str = "data/numina.jsonl",
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    tokenizer: str = None,
    output_path: str = "results/numina/1.jsonl",  # path to .jsonl output path
    chunk_limit: int = 8192,  # save every chunk_limit generations until finished or preempted; when job resumes, saved generations will be reloaded and continue from there.
    tensor_parallel_size: int = 1,  # up to max GPUs in the machine,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    seed: int = None,
    max_tokens: int = 32768,
    n_repeat: int = 16,
    ) -> List[Dict]:
    """
    Use Vllm to generate on a jsonl file


    Args:
        prompt_path (str): path to the .jsonl file
        model (str): can be any of the following:
            (1) any model names from huggingface;
            (2) path to huggingface model dir;
            (3)pre-defined model_names from a `MODEL_NAMES_TO_PATHS` in `ram.internal_constants`.
        output_path (str): file path for storing final generations. If empty, generations will be saved to output_path will be dump_dir / "generations.jsonl".
            NOTE: Either use dump_dir or output_path to store the results. No need to set both.
            NOTE: Recommend to use dump_dir over output_path. output_path is left empty when running slurm grid sweep.
        chunk_limit (int): max size of prompts for vllm lm engine at a time, outputs will be saved as intermediate files
            when job resumes, saved generations will be reloaded and continue from there if reuse_generated is True.
        tensor_parallel_size (int): default is 4 according to Ping's experiences.
        frequency_penalty (float): Float that penalizes new tokens based on their frequency in the generated text so far.
            Values > 0 encourage the model to use new tokens, while values < 0 encourage the model torepeat tokens.
        repetition_penalty (float): penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
            Warning: difference with xlformers (where value<=1 is no penalty) here it should NOT go below 1.
        temperature (float): controls the randomness of the sampling. Zero means greedy sampling.
        top_p (float): controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k (int): Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
        seed (int): Random seed to use for the generation. Default is None
        max_tokens (int): Maximum number of tokens to generate per output sequence. Default is 2048


    Returns:
        List[Dict]: output set. given input ex, output
            {
                **ex,
                vllm_output': {
                    'output': vllm_output.text,
                    'token_ids': vllm_output.token_ids,
                    'cumulative_logprob': vllm_output.cumulative_logprob,
                    'finish_reason': vllm_output.finish_reason,
                },
            }
    """
    assert (
        frequency_penalty >= 0
    ), "frequency_penalty < 0 in vllm means ENCOURAGE model to repeat"
    assert (
        repetition_penalty >= 1
    ), "repetition_penalty < 1 in vllm means ENCOURAGE model to repeat"
    assert (
    #    output_path is not None or dump_dir is not None
        output_path is not None
    ), "must set output_path or dump_dir"


    # load data
    with open(prompt_path) as f:
    #    data = [json.loads(l) for l in f.read()]
        data = [json.loads(line) for line in f]

    data = load_dataset("json", data_files=prompt_path, split="train")
    print(f"Number of data: {len(data)}")
    # print(data[0]['prompt'])
    # print(stop)
    tok = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model)
    # data = [format_prompt(ex, tok) for ex in data]
    data = data.repeat(n_repeat)
    data = data.map(
        format_prompt,
        fn_kwargs={
            "tokenizer": tok,
        },
        remove_columns=[c for c in data.column_names if c not in {"input", "solution"}],
        batched=False,
        num_proc=1
    )
    print(f"Number of prompts: {len(data)}")
    data = list(data)
    print(data[0]['input'])
    
    logger.info(f"Loaded {len(data)} prompts from {prompt_path}")
    print(f"Loaded {len(data)} prompts from {prompt_path}")
    # initialization
    llm = VllmInference(
        model_name=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
    )
    logger.info(f"Start evaluating")
    bos_token_id = llm.vllm_model.get_tokenizer().bos_token_id
    out_data = []
    for cur_chunk in range(0, math.ceil(len(data) / chunk_limit)):
        chunk = data[cur_chunk * chunk_limit : (cur_chunk + 1) * chunk_limit]
        
        prompts = [ex["input"] for ex in chunk]
        # prompts = chunk["input"]
        assert len(prompts) == len(chunk)
        outputs = llm.generate(prompts)
        assert len(chunk) == len(outputs)
        for ex, output in zip(chunk, outputs):
            ex = update_serialized_vllm_output(ex, output)
            # # BOS sanity check
            # if ex["vllm_output"]["prompt_token_ids"][0] != bos_token_id:
            #     raise AssertionError("No BOS token detected in tokenized prompt")
            # elif (
            #     len(ex["vllm_output"]["prompt_token_ids"]) > 1
            #     and ex["vllm_output"]["prompt_token_ids"][1] == bos_token_id
            # ):
            #     raise AssertionError("Double BOS token detected in tokenized prompt")
        out_data.extend(chunk)


    destroy_model_parallel()
#    del llm.vllm_model.llm_engine.model_executor.driver_worker
#    del llm.vllm_model  # Isn't necessary for releasing memory, but why not

    try:
        import ray
        ray.shutdown()
    except ImportError:
        pass

    del llm                           # let Python collect everything
    gc.collect()
    torch.cuda.empty_cache()


    return out_data




if __name__ == "__main__":
#    fire.Fire(vllm_generate)
    prompt_path = "data/numina_1k.jsonl"
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--prompt_path",
        type=str,
        default=prompt_path,
        help="Path to the input prompt file",
    )
    argparse.add_argument(
        "--model",
        type=str,
        default=model,
        help="Model name or path to the model directory",
    )
    argparse.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs to use for tensor parallelism",
    )
    argparse.add_argument(
        "--output_path",
        type=str,
        default="results/numina_1k_res_1.jsonl",
        help="Path to the output file",
    )
    argparse.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate",
    )
    args = argparse.parse_args()
    # out = vllm_generate(
    #     prompt_path=args.prompt_path,
    #     model=args.model,
    #     output_path=args.output_path,
    #     chunk_limit=8192,
    #     tensor_parallel_size=args.tensor_parallel_size,
    #     frequency_penalty=0.0,
    #     repetition_penalty=1.0,
    #     temperature=0.6,
    #     top_p=0.95,
    #     top_k=20,
    #     min_p=0,
    #     seed=42,
    #     max_tokens=args.max_tokens,
    # )
    out = vllm_generate(
        prompt_path=args.prompt_path,
        model=args.model,
        output_path=args.output_path,
        chunk_limit=8192,
        tensor_parallel_size=args.tensor_parallel_size,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1,
        top_k=1,
        min_p=0.0,
        seed=42,
        max_tokens=args.max_tokens,
    )
    ### Save the output to a file
    with open(args.output_path, "w") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
