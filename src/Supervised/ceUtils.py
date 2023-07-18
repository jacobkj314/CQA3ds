import inspect
import sys
from hparams import *
print(f'lam_mle={lam_mle}\nlam_ce={lam_ce}')

# # # Setup CEloss
import torch
def get_first_token_likelihood_from_logits(out_ids, logits=None):
    softmaxedScores = torch.log(torch.softmax(logits,dim=1)) # softmax and transform to log-likelihood
    scores = softmaxedScores[range(out_ids.shape[0]),out_ids[:,0]] #get the likelihood for the associated token and instance
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
def bundling(logits, labels, bundle_size):
    index = 0
    for size in bundle_size:
        yield logits[index:index+size], labels[index:index+size]
        index += size
def bundling_old(batch): 
    batch_size = batch['input_ids'].shape[0]
    for i in range(batch_size):
        yield {col:batch[col][i, ...] for col in batch}


# # # START TRAINER SETUP CODE
import torch
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
class Seq2SeqTrainerCE(Seq2SeqTrainer):
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # # #Need to prevent bundle_size from getting to model.forward
        bundle_size = inputs.pop("bundle_size")
        # # #

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

        if lam_ce != 0: # # # save a bit of computation if we are not using ce (maybe)
            ce_losses = []
            for bundle_logits, bundle_labels in bundling(outputs['logits'], inputs['labels'], bundle_size):
                ce_losses.append(ce_loss_fn(bundle_logits, bundle_labels))       
        ce_loss = (sum(ce_losses) / len(ce_losses)) if (lam_ce != 0) else (0)

        self.log({'mle_loss':mle_loss.item(), 'ce_loss':ce_loss.item()}) #There seem to be issues checkpointing if I try to log using objects that have backward hooks in the computation graph which prevent them from calling __deepcopy__()
        loss = lam_mle * mle_loss + lam_ce * ce_loss 
        # # # END OF MY NEW CODE

        return (loss, outputs) if return_outputs else loss
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys()) + ["bundle_size"] # # # I need to make this an expected column so it doesn't get removed!!
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))


    from torch import nn
    from typing import Dict, Union, Any, Optional, List, Tuple
    from transformers.deepspeed import is_deepspeed_zero3_enabled

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        # # # We don't care about the bundle_size if we are not trying to backprop
        if 'bundle_size' in inputs.keys():
            inputs.pop('bundle_size')
        # # #
        
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys
        )





class DataCollatorForSeq2SeqCE(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        for feature in features: # # # This is also new. The padding is ending up weird in the labels column
            labels = [label for label in feature['labels'] if isinstance(label, list)]
            if labels: # # # But if there aren't labels (i.e. at inference), we don't need to pass the labels
                feature['labels'] = labels

        '''
        This is new - I am having it join several bundles into a single batch
        This is also some of the more cursed python I've ever written, so as an explanation:
            is_bundled checks whether the input_ids for each instance in the dataset looks like [. . .] or like [[. . .] . . . [. . .]] that is, whether the dataset is bundled
            if it is_bundled:
                we use sum() to join together several lists of lists into one big list of lists. This requires passing the extra parameter [] 
            otherwise:
                we use list() to join together several         lists into one big list of lists
        '''
        features = [ 
            {
                colname: 
                    (   #Which function to use for aggregating
                        sum 
                        if (is_bundled := isinstance(features[0]['input_ids'][0], list)) else 
                        list
                    )(
                        *
                        (   
                            (
                                lambda param : #Which parameters to pass to the function
                                (
                                    (param, [])
                                    if is_bundled else
                                    (param,)
                                )
                            )(
                                (feature[colname] for feature in features)
                            )
                        )
                    ) for colname in features[0]
            }
        ]

        bundle_size = features[0].pop('bundle_size') if 'bundle_size' in features[0].keys() else None# # # bundle_size doesn't need to be passed to tokenizer for padding/tensorizing

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        features = { # # # This is also new - need to lose the outermost layer of the tensor that is added when padding again - but only when assembling from bundles and not from instances
            colname : features[colname][0] if len(features[colname].shape) == 3 else features[colname] for colname in features
        }
        if bundle_size is not None:
            features['bundle_size'] = bundle_size # # # add bundle_size back on (if we care to track it)

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
    


