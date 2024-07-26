import torch
from transformers import AutoTokenizer, AutoConfig
import yaml
import gradio as gr
from src.MultiLabelBert import BertForMultilabelClassification

# Load configuration parameters
with open('config/param.yaml', 'r') as file:
    param = yaml.safe_load(file)

def initiate_model():
    """
    Load and initialize the tokenizer, model, and configuration.
    model_name is fetched from param.yaml

    return: tokenizer, model, config
    """
    tokenizer = AutoTokenizer.from_pretrained(param['model_name'])
    config = AutoConfig.from_pretrained(param['model_name'])
    model = BertForMultilabelClassification.from_pretrained(param['model_name']) 
    return tokenizer, model, config

# Initialize the model and configurations
tokenizer, model, config = initiate_model()
id2label_sub = config.id2label_subprod
id2label_prod = config.id2label_prod

def classify_complaint(user_input):
    """
    Classify the user complaint and return the category labels.
    """
    # Tokenize user input
    tok_input = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_input = {key: value.to(device) for key, value in tok_input.items()} # moving each data into device GPU/CPU
    
    # Model evaluation
    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(**tok_input)
        subprod_logits, prod_logits = outputs.logits
    
    # Get category labels
    sub_prod_cat = str(int(torch.argmax(subprod_logits, dim=-1).cpu().item()))
    prod_cat = str(int(torch.argmax(prod_logits, dim=-1).cpu().item()))

    # Return the results
    #return f"The complaint should be routed to Product: {id2label_prod[prod_cat]} Section  -> Sub-Product {id2label_sub[sub_prod_cat]}"

    result = f"""
    <div style='font-size: 1.2em; line-height: 1.5;'>
        <strong>Complaint To Be Redirected To:</strong><br> 
        <strong>Product Category:</strong> {id2label_prod[prod_cat]}<br>
        <strong>Sub-Product Category:</strong> {id2label_sub[sub_prod_cat]}
    </div>
    """
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=classify_complaint,
    inputs=gr.Textbox(lines=5, placeholder="Enter User Complaint to Classify",label="User Complaint"),
    outputs=gr.HTML(),
    title="User Complaint Classifier",
    description="Enter a user complaint and classify it into appropriate categories for quick resolution"
)

if __name__ == "__main__":
    iface.launch()
