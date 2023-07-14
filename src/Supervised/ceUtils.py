from hparams import *
print(f'lam={lam}')

# # # Setup CEloss
import torch
def get_first_token_likelihood_from_logits(out_ids, logits=None):
    softmaxedScores = torch.log(torch.softmax(logits,dim=1)) # softmax and transform to log-likelihood
    scores = softmaxedScores[range(out_ids.shape[0]),out_ids[:,1]] #get the likelihood for the associated token and position
    return scores

def ce_loss_fn(lm_logits, labels):
    squeezed_logits = lm_logits[:,0,:] # squeeze away the token dimension, since we are only looking at the next token (yes/no/don't)
    ce = []
    for i in range(labels.shape[0]):#iterate across number of individual samples in bundle
        ce.append(
          get_first_token_likelihood_from_logits(
            labels,
            squeezed_logits.roll(shifts=i, dims=0)  # # # pre-squeeze the logits and then roll between instances
          ) 
        )
    z = torch.log( #normalizing denominator
      sum(torch.exp(term) for term in ce) #add up all the denominators - using regular python sum because they are tensors in a list
    ) 
    ceLoss = torch.mean(ce[0] - z) * -1 #Multiply by -1 so that by minimizing loss we maximize the proportion of the distribution is taken up by the correct answer
    return ceLoss


# # #Setup Bundling
def bundling(batch): 
    batch_size = batch['input_ids'].shape[0]
    for i in range(batch_size):
        yield {col:batch[col][i, ...] for col in batch}


# # # START TRAINER SETUP CODE
import torch
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import Seq2SeqTrainer
class Seq2SeqTrainerCE(Seq2SeqTrainer):
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # # # THE ORIGINAL CODE HERE
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                mle_loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                mle_loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            mle_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # # # END OF THE ORIGINAL CODE
        # # # MY NEW CODE
        mle_losses = []
        ce_losses = []

        for bundle in bundling(inputs):
            ce_losses.append(ce_loss_fn(outputs['logits'], bundle['labels']))
            '''# # # # # 
            if self.label_smoother is not None and "labels" in bundle:
                labels = bundle.pop("labels")
            else:
                labels = None
            outputs = model(**bundle)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    mle_losses.append(self.label_smoother(outputs, labels, shift_labels=True)) # # # mle_loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    mle_losses.append(self.label_smoother(outputs, labels)) # # # mle_loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(bundle.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                mle_losses.append(outputs["loss"] if isinstance(outputs, dict) else outputs[0]) # # # mle_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            '''# # # # # 
        

        # # # # # mle_loss = sum(mle_losses) / len(mle_losses)
        ce_loss = sum(ce_losses) / len(mle_losses)

        self.log({'mle_loss':mle_loss.item(), 'ce_loss':ce_loss.item()}) #There seem to be issues checkpointing if I try to log using objects that have backward hooks in the computation graph which prevent them from calling __deepcopy__()
        loss = mle_loss + lam * ce_loss 
        # # # END OF MY NEW CODE

        return (loss, outputs) if return_outputs else loss
# # # END TRAINER SETUP CODE