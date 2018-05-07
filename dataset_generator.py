import tensorflow as tf


class DatasetGenerator:

    @classmethod
    def generate(cls, max_text_length: int, max_named_entity_size, utterances: set):
        def gen():
            empty_entities = [0 for _ in range(max_named_entity_size)]

            for utterance in utterances:
                length = len(utterance)
                masks = [1 if i < length else 0 for i in range(max_text_length)]

                tokens = [utterance.encoded_tokens[i] if i < length else 0 for i in range(max_text_length)]
                labels = [utterance.encoded_labels[i] if i < length else 0 for i in range(max_text_length)]

                entities = [
                    [entity[i] if i < len(entity) else 0 for i in range(max_named_entity_size)]
                    for entity in utterance.encoded_entities
                ]
                entities = [entities[i] if i < length else empty_entities for i in range(max_text_length)]

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
                    'text': tf.TensorShape([max_text_length]),
                    'named_entity': tf.TensorShape([max_text_length, max_named_entity_size]),
                    'length': tf.TensorShape([]),
                    'mask': tf.TensorShape([max_text_length])
                },
                tf.TensorShape([max_text_length])
            )
        )

        return dataset
