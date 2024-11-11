import fire
from train import STR2MODELCLASS
import json

import logging
logging.basicConfig(level=logging.ERROR)

VERBOSE = True

def main(
        eval_filename: str, output_filename: str,
        base_model_name: str = "meta-llama/Llama-2-7b-hf", lora_checkpoint_dir: str = "",
        # prompt_template_name: str = "base",
        prompt_template_name: str = "",
        do_sample: bool = True, top_p: float = 0.8, top_k: int = 20, 
        num_beams: int = 4, temperature: float = 1.0, max_new_tokens: int = 128,
        batch_size: int = 8,
    ):
    print(f'Evaluating "{eval_filename}" using {base_model_name} with lora weights - {lora_checkpoint_dir}')
    model_class = [v for k, v in STR2MODELCLASS.items() if k in base_model_name]
    if len(model_class) == 0:
        raise ValueError(f"Unknown model name: {base_model_name}")
    elif len(model_class) > 1:
        raise ValueError(f"Multiple model names: {model_class}")
    else:
        print(f"Fine-tuning {base_model_name} with LoRA using class {model_class} ...")
        model_class = model_class[0]

    print(f'Decoding parameters: do_sample: {do_sample}, top_p: {top_p}, top_k: {top_k}, num_beams: {num_beams}, temperature: {temperature}, max_new_tokens: {max_new_tokens}, batch_size: {batch_size}')
    model = model_class(
        base_model=base_model_name,
        prompt_template_name=prompt_template_name,
    )
    predictions = model.predict(
        input_file=eval_filename,
        lora_adapter=lora_checkpoint_dir,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        kshot=False,
        text_output_only=False,
        verbose=VERBOSE,
        batch_size=batch_size,
    )

    with open(output_filename, 'w') as outfile:
        json.dump(predictions, outfile)

if __name__ == "__main__":
    fire.Fire(main)

# def predict(self,
#                 input_file: str = "",
#                 lora_adapter: str = "",
#                 # For ICL
#                 kshot: int = 0,
#                 demo_file: Union[None, str] = None,
#                 # Decoding
#                 do_sample: bool = True,
#                 top_p: float = 0.8,
#                 top_k: int = 20,
#                 num_beams: int = 4,
#                 temperature: float = 1.0,
#                 max_new_tokens: int = 128,
#                 # For classification
#                 label_set: Union[None, List[str]] = None, # A list of labels
#                 verbose: bool = False,
#                 ):