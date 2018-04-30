import json
import os
import shutil

import tensorflow as tf

from dataset_generator import DatasetGenerator
from utterance import Utterance
from vocabulary import Vocabulary


class SlotTagger:

    def __init__(self, bot_name: str, max_length: int):
        self.__bot_name = bot_name
        self.__max_length = max_length
        self.__text_vocab = None
        self.__label_vocab = None
        self.__estimator = None

        model_path = os.path.join(os.path.dirname(__file__), './model', self.__bot_name)
        if os.path.exists(model_path):
            with open(os.path.join(model_path, 'text.vocab'), encoding='utf-8') as fp:
                self.__text_vocab = Vocabulary(json.loads(fp.read()))

            with open(os.path.join(model_path, 'label.vocab'), encoding='utf-8') as fp:
                self.__label_vocab = Vocabulary(json.loads(fp.read()))

            self.__estimator = self.__make_estimator(model_path, len(self.__label_vocab), len(self.__text_vocab))

    @classmethod
    def __make_estimator(cls, model_path, output_size, vocab_size):
        return tf.estimator.Estimator(
            model_fn=cls.__model_fn,
            model_dir=model_path,
            config=tf.estimator.RunConfig(
                save_summary_steps=5,
                save_checkpoints_steps=5,
            ),
            params={
                'cell_size': 200,
                'char_embedding_size': 200,
                'output_size': output_size,
                'vocab_size': vocab_size,
                'learning_rate': 0.0001
            }
        )

    @staticmethod
    def __input_fn(dataset: tf.data.Dataset):
        return dataset.make_one_shot_iterator().get_next()

    @staticmethod
    def __model_fn(features, labels, mode, params):
        cell_size = params['cell_size']
        output_size = params['output_size']
        vocab_size = params['vocab_size']
        embedding_size = params['char_embedding_size']
        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.estimator.ModeKeys.TRAIN else 0.5

        ids = features['ids']
        length = features['length']
        mask = features['mask']

        char_embeddings = tf.get_variable(
            name='char_embeddings',
            shape=[vocab_size, embedding_size],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        inputs = tf.nn.embedding_lookup(char_embeddings, ids)

        def rnn_cell(cell_size):
            cell = tf.nn.rnn_cell.GRUCell(cell_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(cell_size),
            cell_bw=rnn_cell(cell_size),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32
        )

        outputs = outputs[0] + outputs[1]

        outputs = tf.layers.dense(
            inputs=outputs,
            units=output_size,
            activation=tf.nn.relu
        )

        predictions = tf.argmax(outputs, 2)

        loss = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(labels, output_size, dtype=tf.float32),
                logits=outputs,
                weights=mask
            )

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels, predictions, mask)
            }

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=10,
                decay_rate=0.96
            )

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    @classmethod
    def train(cls, bot_name, max_length):
        text_vocab = Vocabulary()
        label_vocab = Vocabulary()
        utterances = Utterance.fetch(bot_name)

        converted = []
        for utterance in utterances:
            tokens = list(map(lambda token: text_vocab.transform(token), utterance.tokens()))
            labels = list(map(lambda label: label_vocab.transform(label), utterance.labels()))
            converted.append({
                'tokens': tokens,
                'labels': labels
            })

        model_path = os.path.join(os.path.dirname(__file__), './model', bot_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path)

        with open(os.path.join(model_path, 'text.vocab'), mode='w', encoding='utf-8') as fp:
            fp.write(text_vocab.save())

        with open(os.path.join(model_path, 'label.vocab'), mode='w', encoding='utf-8') as fp:
            fp.write(label_vocab.save())

        estimator = cls.__make_estimator(model_path, len(label_vocab), len(text_vocab))

        dataset = DatasetGenerator.generate(max_length, converted)
        dataset = dataset.shuffle(1000).repeat(1000).batch(100)

        estimator.train(
            lambda: cls.__input_fn(dataset)
        )

        result = estimator.evaluate(
            lambda: cls.__input_fn(dataset)
        )

        print(result)

    def tag(self, utterance: Utterance):
        tokens = list(map(lambda token: self.__text_vocab.transform(token), utterance.tokens()))
        labels = list(map(lambda label: self.__label_vocab.transform(label), utterance.labels()))
        converted = {
            'tokens': tokens,
            'labels': labels
        }

        dataset = DatasetGenerator.generate(self.__max_length, [converted])
        dataset = dataset.batch(1)

        predictions = self.__estimator.predict(
            lambda: self.__input_fn(dataset)
        )

        prediction = list(predictions)[0][:len(tokens)]
        labels = list(map(lambda num: self.__label_vocab.restore(num), prediction))
        print(utterance.tokens())
        print(labels)
