import tensorflow as tf

from named_entity import NamedEntity

tf.logging.set_verbosity(tf.logging.INFO)


class SlotTagger:

    def __init__(self, model_path: str, text_vocab_size: int, ne_vocab_size: int, output_size: int,
                 max_length: int, hyper_params: dict):
        self.__estimator = tf.estimator.Estimator(
            model_fn=self.__model_fn,
            model_dir=model_path,
            config=tf.estimator.RunConfig(
                save_checkpoints_steps=50,
                log_step_count_steps=10,
                keep_checkpoint_max=1
            ),
            params={
                'cell_size': hyper_params['cell_size'],
                'text_embedding_size': hyper_params['text_embedding_size'],
                'ne_embedding_size': hyper_params['ne_embedding_size'],
                'text_vocab_size': text_vocab_size,
                'ne_vocab_size': ne_vocab_size,
                'max_length': max_length,
                'output_size': output_size,
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
        text_vocab_size = params['text_vocab_size']
        ne_vocab_size = params['ne_vocab_size']
        text_embedding_size = params['text_embedding_size']
        ne_embedding_size = params['ne_embedding_size']
        max_length = params['max_length']
        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.estimator.ModeKeys.TRAIN else 0.5

        text = features['text']
        named_entity = features['named_entity']
        length = features['length']
        mask = features['mask']

        text_embeddings = tf.get_variable(
            name='text_embeddings',
            shape=[text_vocab_size, text_embedding_size],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        ne_embeddings = tf.get_variable(
            name='ne_embeddings',
            shape=[ne_vocab_size, ne_embedding_size],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        text = tf.nn.embedding_lookup(text_embeddings, text)
        named_entity = tf.nn.embedding_lookup(ne_embeddings, named_entity)
        named_entity = tf.reshape(named_entity, [-1, max_length, NamedEntity.SIZE * ne_embedding_size])

        inputs = tf.concat([text, named_entity], axis=2)

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

            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
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
    def train(cls, model_path: str, text_vocab_size: int, ne_vocab_size: int, output_size: int, max_length: int,
              hyper_params: dict, dataset: tf.data.Dataset):
        slot_tagger = SlotTagger(model_path, text_vocab_size, ne_vocab_size, output_size, max_length, hyper_params)
        slot_tagger.__estimator.train(lambda: cls.__input_fn(dataset), steps=201)
        result = slot_tagger.__estimator.evaluate(lambda: cls.__input_fn(dataset), steps=1)
        print(result)

        return slot_tagger

    def predict(self, dataset: tf.data.Dataset):
        predictions = self.__estimator.predict(
            lambda: self.__input_fn(dataset)
        )

        return list(predictions)
