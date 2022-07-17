from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_metric

from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer

import pandas as pd
import hazm
import torch


class ClassificationModel:

    def __init__(self, model_name, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = None
        self.label2id = None

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.classifier = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    @staticmethod
    def training_args_builder(output_dir="../models/", learning_rate=2e-3, train_batch_size=32,
                              eval_batch_size=32, num_train_epochs=30, weight_decay=0.01):
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            save_steps=1_000
        )

    def train(self, dataset, training_args=None):
        training_args = training_args or self.training_args_builder()

        train_df, validation_df = train_test_split(
            dataset, test_size=0.2, random_state=42, shuffle=True)

        train_dataset, validation_dataset = \
            Dataset.from_dict(train_df), Dataset.from_dict(validation_df)

        dataset = DatasetDict({"train": train_dataset, "test": validation_dataset})

        def preprocess_function(data):
            return self.tokenizer(data["text"], truncation=True, padding=True)

        tokenized_data = dataset.map(
            preprocess_function, batched=True, remove_columns=['text'])

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        optimizer = Adafactor(
            self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)

        Trainer(
            eval_dataset=tokenized_data["test"],
            train_dataset=tokenized_data["train"],
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
        ).train()

    def predict(self, texts):
        self.classifier.model.to('cpu')
        inner_labels = self.classifier(texts)
        return list(map(lambda lab: int(lab['label'].split('_')[1]), inner_labels))


if __name__ == '__main__':
    normalizer = hazm.Normalizer(token_based=True)
    df = pd.read_csv('../../resources/shahnameh-dataset.csv')

    df['text'] = df.text.apply(normalizer.normalize)

    labels = set(df.labels.unique())
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    df['labels'] = df.labels.apply(label2id.get)

    main_df, test_df = train_test_split(
        df, test_size=0.1, random_state=42, shuffle=True)

    main_df = main_df.groupby('labels').sample(
        main_df.groupby('labels').size().mean(), replace=True)

    directory = None
    classifier = ClassificationModel(
        directory or "HooshvareLab/bert-fa-zwnj-base", len(labels))

    classifier.train(main_df, None)
    predictions = classifier.predict(test_df['text'].to_list())

    metric = load_metric("f1")
    print("f1_score: ", metric.compute(
        predictions=predictions, references=test_df['labels'], average='micro'))

    metric = load_metric("accuracy")
    print("accuracy: ", metric.compute(
        predictions=predictions, references=test_df['labels']))

    titles = list(map(id2label.get, predictions))
