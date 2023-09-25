from os.path import join as pathjoin


MILA_WEIGHTS = "/network/weights"
BASE_LLAMA2_WEIGHTS = pathjoin(MILA_WEIGHTS, "/llama.var/llama2")


def llama2weights(name):
    return pathjoin(BASE_LLAMA2_WEIGHTS, name)


LLAMA_WEIGHTS = {
    "7b":       llama2weights("llama-2-7b-chat"),
    "7b-chat":  llama2weights("llama-2-7b-chat"),
    "13b":      llama2weights("llama-2-13b"),
    "13b-chat": llama2weights("llama-2-13b-chat"),
    "70b":      llama2weights("llama-2-70b"),
    "70b-chat": llama2weights("llama-2-70b-chat"),
}

TOKENIZER_CHK = pathjoin(BASE_LLAMA2_WEIGHTS, "tokenizer_checklist.chk")
TOKENIZER = pathjoin(BASE_LLAMA2_WEIGHTS, "tokenizer.model")


def get_llama_params(model):
    return dict(
        ckpt_dir=LLAMA_WEIGHTS[model],
        tokenizer_path=TOKENIZER
    )


if __name__ == "__main__":

    for k, v in LLAMA_WEIGHTS.items():
        print(v)