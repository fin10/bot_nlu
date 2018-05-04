import tensorflow as tf


class DatasetGenerator:

    @staticmethod
    def generate(max_length: int, utterances: set):
        def gen():
            for utterance in utterances:
                length = len(utterance)
                masks = [1 if i < length else 0 for i in range(max_length)]

                tokens = [utterance.encoded_tokens[i] if i < length else 0 for i in range(max_length)]
                labels = [utterance.encoded_labels[i] if i < length else 0 for i in range(max_length)]

                yield ({
                            'ids': tokens,
                            'length': length,
                            'mask': masks
                        }, labels)

        dataset = tf.data.Dataset.from_generator(
            gen,
            (
                {
                    'ids': tf.int32,
                    'length': tf.int32,
                    'mask': tf.int32
                },
                tf.int32
            ),
            (
                {
                    'ids': tf.TensorShape([max_length]),
                    'length': tf.TensorShape([]),
                    'mask': tf.TensorShape([max_length])
                },
                tf.TensorShape([max_length])
            )
        )

        return dataset
