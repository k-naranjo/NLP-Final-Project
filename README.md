# NLP-Final-Project
## Next, let's load the human evaluation data


TG_human = load_dataset("skg/toxigen-data", name="annotations", token="") #input your token here
human_eval = pd.DataFrame(TG_human["train"])

print(human_eval.shape)
human_eval.head()
#print(human_eval["Input.prompt"])
#print(human_eval["Input.text"])
import re
def clean_text(text):
    text = re.sub(r'\\n', ' ', text)  
    text = text.replace('-', '').strip()  
    text = re.sub(r'\s+', ' ', text) 
    text = text.replace('\\','').strip() 
    return text
human_eval['Input.prompt'] = human_eval['Input.prompt'].astype(str).str.replace("b'", "").str.replace("'", "")
human_eval['Input.prompt'] = human_eval['Input.prompt'].apply(clean_text)
print(human_eval['Input.prompt'])
print(human_eval.loc[0,'Input.prompt'])
