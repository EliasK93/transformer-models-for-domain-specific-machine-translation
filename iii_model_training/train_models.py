"""
See https://github.com/sravan1320/NMT/blob/main/fine_tune_hugging_face_translation_model.ipynb
"""
from datasets import load_dataset, load_metric
import datasets
import numpy as np
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import model_utils


metric_sacrebleu = load_metric("sacrebleu")


def load_raw_datasets() -> datasets.DatasetDict:
    """
    Loads train, validation and test set from local JSONL files and wraps them in a DatasetDict.

    :return: DatasetDict containing the sets
    """
    train_dataset = list(load_dataset('json', data_files='training_data/jsonl/train.json').values())[0]
    val_dataset = list(load_dataset('json', data_files='training_data/jsonl/val.json').values())[0]
    test_dataset = list(load_dataset('json', data_files='training_data/jsonl/test.json').values())[0]
    return datasets.DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})


def preprocess_function(examples, source_lang="en", target_lang="de", prefix="",
                        max_input_length=128, max_target_length=128):
    """
    See https://github.com/sravan1320/NMT/blob/main/fine_tune_hugging_face_translation_model.ipynb
    """
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    """
    See https://github.com/sravan1320/NMT/blob/main/fine_tune_hugging_face_translation_model.ipynb
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, metric=metric_sacrebleu):
    """
    See https://github.com/sravan1320/NMT/blob/main/fine_tune_hugging_face_translation_model.ipynb
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def preprocess_datasets(raw_datasets_: datasets.DatasetDict, model_type_: str):
    """
    Applies the preprocess function to a DatasetDict and returns a DatasetDict containing
    the preprocessed and tokenized sets.

    :param raw_datasets_: DatasetDict of the raw sets
    :param model_type_: name of the model (model_id in huggingface)
    :return: DatasetDict containing the preprocessed and tokenized sets
    """
    return raw_datasets_.map(preprocess_function, batched=True,
                             fn_kwargs={"prefix": model_utils.prefixes[model_type_]})


if __name__ == '__main__':

    # LOAD DATASET
    raw_datasets = load_raw_datasets()

    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-base"]:

        # LOAD MODEL
        tokenizer, model = model_utils.load_model(model_type=model_type, local=False)

        # PREPROCESS
        tokenized_datasets = preprocess_datasets(raw_datasets, model_type_=model_type)

        # TRAIN
        output_dir = model_utils.model_dirs[model_type]
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir, overwrite_output_dir=False, seed=1, num_train_epochs=10, learning_rate=2e-5,
            weight_decay=0.01, save_total_limit=2, per_device_train_batch_size=model_utils.batch_sizes[model_type],
            per_device_eval_batch_size=model_utils.batch_sizes[model_type], predict_with_generate=True,
            evaluation_strategy="epoch", save_strategy="epoch"
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(model, args, tokenizer=tokenizer, train_dataset=tokenized_datasets["train"],
                                 eval_dataset=tokenized_datasets["validation"], data_collator=data_collator,
                                 compute_metrics=compute_metrics)
        trainer.train()
