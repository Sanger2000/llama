# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from long_prompt import prompt

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator, tokenizer


def main(
    generator,
    tokenizer,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 1,
    prompt_len: int = 512,
    max_batch_size: int = 32,
    max_gen_len: int = 256,
):

    raw_tokens = tokenizer.encode(prompt, bos=False, eos=False)[:prompt_len]
    new_prompt = tokenizer.decode(raw_tokens)

    prompts = [
        new_prompt for _ in range(max_batch_size)
    ]
    results = generator.generate(
        prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )

    for result in results:
        #print(result)
        #print("\n==================================\n")
        pass

def other_main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 1,
    prompt_len: int = 512,
    max_gen_len: int = 256,
    ):

    max_seq_len = prompt_len + max_gen_len + 5
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # calculate the actual max batch size:
    highest_possible = 170 * 1000 / (max_seq_len * 2.6)
    for possible_size in range(0, int(highest_possible), 16):
        if possible_size > highest_possible:
            break
        else:
            max_batch_size = possible_size

    #max_batch_size = 24

    print('max batch size', max_batch_size)
    generator, tokenizer = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    for i in (4, 4, 1, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128):
        if i > max_batch_size:
            break
        # We warmup initially
        print('batch size', i)
        main(
        generator,
        tokenizer,
        ckpt_dir,
        tokenizer_path,
        temperature,
        top_p,
        prompt_len,
        i,
        max_gen_len,
        )



if __name__ == "__main__":
    #fire.Fire(main)
    fire.Fire(other_main)
