# Bot NLU

Bot NLU is a tiny NLU engine for bot program. It only performs the slot filling among SLU tasks using a bidirectional RNN with TensorFlow.

## Requirements

- Python 3
- Tensorflow 1.8
- MongoDB
- pyokt
- pymongo
- spans
- flask
- python-dotenv

## Usage

There are 3 steps to use Bot NLU such belows.

- Submission
- Training
- Prediction

### Submission

All of contents for bot definitions should be stored on mongoDB. Submission is a step to upload contents on mongoDB.

The followings are files which should be included in submission.

- config.json (*required*)
  - name: Name of bot.
  - hyper_params: Hyper parameters which is used to train model.
    - batch_size: Number of samples on ML training step. 
    - steps: Number of steps which to train model.
    - cell_size: Number of units in the RNN cell.
    - text_embedding_size: Dimension of embedding matrix for text.
    - ne_embedding_size: Dimension of embedding matrix for named entity.

  ```json
  {
    "name": "hello_bot",
    "hyper_params": {
      "batch_size": 100,
      "steps": 501,
      "cell_size": 50,
      "text_embedding_size": 50,
      "ne_embedding_size": 50
    }
  }
  ```

- training (*required*)
  - It's a list of utterances which is used to train model. () means text of slot. And [] means type of slot.

  ```text
  show me (chinese)[foodType] restaurants
  find (spicy)[taste] food
  find restaurants nearby (new york)[location] city
  ```

- {named_entity}.ne (*optional*)
  - Define this file if you need extra information on training. If there is an utterance which has text which matches text from this file, it will be used extra information on training. We can define multiple files.

  ```text
  itailan
  chinese
  korean
  seafood
  ...
  ```

When files are ready we can submit them by python script below.

```sh
python3 bot.py submit {path}
```

### Training

Training step performs generating model with a submitted bot. A trained model is stored on **local** by submission time of bot. So model should be generated whenever we submit a bot newly.

We can perform this step by python script below.

```sh
python3 bot.py train {bot_name}
```

### Prediction

It's possible to extract slots from input utterance when trained model for bot is ready.

We can predict slots with utterance by python script below.

```sh
python3 bot.py predict {bot_name} {utterance}
```