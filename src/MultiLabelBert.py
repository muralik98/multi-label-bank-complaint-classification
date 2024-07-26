import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from datasets import ClassLabel

class BertForMultilabelClassification(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_subprod_labels = config.num_subprod_labels
        self.num_prod_labels = config.num_prod_labels
        # Load model body
        self.bert = BertModel(config, add_pooling_layer=True)
        # Classification Head 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subprod_classifier = nn.Linear(config.hidden_size, config.num_subprod_labels )
        self.prod_classifier = nn.Linear(config.hidden_size, config.num_prod_labels)
        # Weight Initialization 
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                subprod_labels=None, prod_labels=None, **kwargs):

        # Model Output 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # Considering the first token for classification purpose 
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        subprod_logits = self.subprod_classifier(pooled_output)
        prod_logits = self.prod_classifier(pooled_output)
        # Calculate losses
        loss = None
        if subprod_labels is not None and prod_labels is not None:
           
            subprod_loss_fct = nn.CrossEntropyLoss()
            prod_loss_fct = nn.CrossEntropyLoss()
            subprod_loss = subprod_loss_fct(subprod_logits.view(-1, self.num_subprod_labels), subprod_labels.view(-1))
            prod_loss = prod_loss_fct(prod_logits.view(-1, self.num_prod_labels), prod_labels.view(-1))
            loss = subprod_loss + prod_loss
        # Return model output object
        return SequenceClassifierOutput(loss=loss, logits=(subprod_logits, prod_logits), 
                                        hidden_states=outputs.hidden_states, 
                                        attentions=outputs.attentions)


