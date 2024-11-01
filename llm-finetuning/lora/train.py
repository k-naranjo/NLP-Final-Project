import fire
from llama_lora import Llama_Lora


import logging

STR2MODELCLASS = {
    "Llama": Llama_Lora
}

logging.basicConfig(level=logging.ERROR)

def main(
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "",
        train_file: str = "",
        val_file: str = "",
        val_set_size: int = 128,
        prompt_template_name: str = "base",
        learning_rate: float = 3e-5,
        num_epochs: int = 2,
        fp16: bool = True,
):
    model_class = [v for k, v in STR2MODELCLASS.items() if k in base_model_name]
    if len(model_class) == 0:
        raise ValueError(f"Unknown model name: {base_model_name}")
    elif len(model_class) > 1:
        raise ValueError(f"Multiple model names: {model_class}")
    else:
        print(f"Fine-tuning {base_model_name} with LoRA using class {model_class} ...")
        model_class = model_class[0]

        model = model_class(
            base_model=base_model_name,
            prompt_template_name=prompt_template_name,
        )
        model.train(
            train_file=train_file,
            val_file=val_file,
            val_set_size=val_set_size,
            output_dir=f"./ckp_{base_model_name.replace('/', '_')}" if not output_dir else output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            group_by_length=False,
            fp16=fp16,
        )

if __name__ == "__main__":
    fire.Fire(main)