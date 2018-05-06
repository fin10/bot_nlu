import tensorflow as tf

from named_entity import NamedEntity


class DatasetGenerator:

    @staticmethod
    def generate(max_length: int, utterances: set):
        def gen():
            for utterance in utterances:
                length = len(utterance)
                masks = [1 if i < length else 0 for i in range(max_length)]

                tokens = [utterance.encoded_tokens[i] if i < length else 0 for i in range(max_length)]
                labels = [utterance.encoded_labels[i] if i < length else 0 for i in range(max_length)]

                empty_entities = [0 for _ in range(NamedEntity.SIZE)]
                encoded_entities = [
                    [entity[i] if i < len(entity) else 0 for i in range(NamedEntity.SIZE)]
                    for entity in utterance.encoded_entities
                ]
                entities = [encoded_entities[i] if i < length else empty_entities for i in range(max_length)]

                yield ({
                    'text': tokens,
                    'named_entity': entities,
                    'length': length,
                    'mask': masks
                }, labels)

        dataset = tf.data.Dataset.from_generator(
            gen,
            (
                {
                    'text': tf.int32,
                    'named_entity': tf.int32,
                    'length': tf.int32,
                    'mask': tf.int32
                },
                tf.int32
            ),
            (
                {
                    'text': tf.TensorShape([max_length]),
                    'named_entity': tf.TensorShape([max_length, NamedEntity.SIZE]),
                    'length': tf.TensorShape([]),
                    'mask': tf.TensorShape([max_length])
                },
                tf.TensorShape([max_length])
            )
        )

        return dataset
