from functools import lru_cache

import flask

from bot import Bot

app = flask.Flask(__name__)


@lru_cache(maxsize=3)
def fetch_bot(bot: str):
    return Bot.fetch(bot)


@app.route('/clear', methods=['POST'])
def clear():
    fetch_bot.cache_clear()
    return 'The cache is cleared.'


@app.route('/<bot>/predict', methods=['POST'])
def predict(bot: str):
    try:
        utterance = flask.request.form['utterance']
        app.logger.debug('utterance: ' + utterance)
        if not utterance or len(utterance) == 0:
            raise AttributeError('There is no utterance.')

        bot = fetch_bot(bot)
        if not bot.is_trained():
            result = bot.train()
            app.logger.debug(result)

        result = bot.predict(utterance)
        return flask.json.jsonify(result)
    except AttributeError as e:
        return flask.make_response(str(e), 404)
    except Exception as e:
        return flask.make_response(str(e), 500)


if __name__ == '__main__':
    app.run(debug=True)
