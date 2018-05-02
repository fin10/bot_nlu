import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class SlotTagger:

    def __init__(self, model_path: str, vocab_size: int, output_size: int, hyper_params: dict):
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
                'char_embedding_size': hyper_params['char_embedding_size'],
                'vocab_size': vocab_size,
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
    def train(cls, model_path: str, vocab_size: int, output_size: int, hyper_params: dict, dataset: tf.data.Dataset):
        slot_tagger = SlotTagger(model_path, vocab_size, output_size, hyper_params)
        slot_tagger.__estimator.train(lambda: cls.__input_fn(dataset), steps=201)
        result = slot_tagger.__estimator.evaluate(lambda: cls.__input_fn(dataset), steps=1)
        print(result)

        return slot_tagger

    def predict(self, dataset: tf.data.Dataset):
        predictions = self.__estimator.predict(
            lambda: self.__input_fn(dataset)
        )

        return list(predictions)
