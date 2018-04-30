import tensorflow as tf


class DatasetGenerator:

    @staticmethod
    def generate(max_length, items: list):
        def gen():
            for item in items:
                tokens = item['tokens']
                labels = item['labels']
                length = len(tokens)

                tokens = [tokens[i] if i < length else 0 for i in range(max_length)]
                labels = [labels[i] if i < length else 0 for i in range(max_length)]
                masks = [1 if i < length else 0 for i in range(max_length)]

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
