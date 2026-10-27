from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# backbone model path
MODEL_MAP = {
    "qwen4B": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen30B": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "ministral3B": "mrfakename/Ministral-3-3B-Instruct-2512-Llamafied-TextOnly",
    "ministral14B": "mrfakename/Ministral-3-14B-Instruct-2512-Llamafied-TextOnly",
}

# lora adapter path
LORA_MAP = {
    "qwen4B": {
        "stage1": "lora_adapter/qwen4B_st1",
        "stage2": "lora_adapter/qwen4B_st2",
        "stage3": "lora_adapter/qwen4B_st3",
        "reflect": "lora_adapter/qwen4B_reflection"
    },
    "qwen30B": {
        "stage1": "lora_adapter/qwen30B_st1",
        "stage2": "lora_adapter/qwen30B_st2",
        "stage3": "lora_adapter/qwen30B_st3",
        "reflect": "lora_adapter/qwen30B_reflection"
    },
    "ministral3B": {
        "stage1": "lora_adapter/ministral3B_st1",
        "stage2": "lora_adapter/ministral3B_st2",
        "stage3": "lora_adapter/ministral3B_st3",
        "reflect": "lora_adapter/ministral3B_reflection"
    },
    "ministral14B": {
        "stage1": "lora_adapter/ministral14B_st1",
        "stage2": "lora_adapter/ministral14B_st2",
        "stage3": "lora_adapter/ministral14B_st3",
        "reflect": "lora_adapter/ministral14B_reflection"
    },
}


# backbone & tokenizer loader
def load_backbone(backbone_path):
    model = AutoModelForCausalLM.from_pretrained(
        backbone_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model

def load_tokenizer(backbone_path):
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


# model loader (attach & detach lora)
def load_trained_backbone(backbone_key):
    backbone_path = MODEL_MAP[backbone_key]
    lora_dirs = LORA_MAP[backbone_key]

    print(f"Load backbone: {backbone_path}")
    base = load_backbone(backbone_path)
    tokenizer = load_tokenizer(backbone_path)

    base.config.use_cache = False
    base.eval()

    return base, tokenizer, lora_dirs

def build_multi_adapter_model(base, lora_dirs):
    model = PeftModel.from_pretrained(base, lora_dirs["stage1"], adapter_name="stage1", is_trainable=False)

    model.load_adapter(lora_dirs["stage2"], adapter_name="stage2")
    model.load_adapter(lora_dirs["stage3"], adapter_name="stage3")
    model.load_adapter(lora_dirs["reflect"], adapter_name="reflect")

    model.eval()
    model.config.use_cache = False
    return model

def set_stage_adapter(model, name):
    model.set_adapter(name)
    model.eval()
    return model


# model loader (prompting)
def load_base_model(backbone_key):
    backbone_path = MODEL_MAP[backbone_key]

    print(f"Load model: {backbone_path}")
    base = load_backbone(backbone_path)
    tokenizer = load_tokenizer(backbone_path)

    base.config.use_cache = False
    base.eval()

    return base, tokenizer