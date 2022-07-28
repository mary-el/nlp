from datetime import datetime

import telebot
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, \
    AutoModelForSeq2SeqLM

TOKEN = 'PUT_TOKEN_HERE'

HELP_MSG = """
Напишите начало вопроса, и бот допишет его и ответит на него
Команды:
[/question (/q)] <начало вопроса> - дописать вопрос и ответить на него
/answer (/a) <вопрос> - ответить на вопрос
/help
____________________________
Write the beginning of the question for the bot to finish and answer it
Commands usage help:
[/question (/q)] <beginning of question> - finish the question and answer it
/answer (/a) <question> - write the full question for the bot to answer it
/help
"""

bot = telebot.TeleBot(token=TOKEN, threaded=False)
keyboard = telebot.types.ReplyKeyboardMarkup()
keyboard.row('/start')

my_model_path = "mary905el/rugpt3large_neuro_chgk"
my_model_path_ans = "mary905el/ruT5_neuro_chgk_answering"

tokenizer_gen = AutoTokenizer.from_pretrained(my_model_path)
model_gen = AutoModelForCausalLM.from_pretrained(my_model_path)

model_ans = AutoModelForSeq2SeqLM.from_pretrained(my_model_path_ans)
tokenizer_ans = T5Tokenizer.from_pretrained(my_model_path_ans)

log_file = 'log.log'


def generate_text(beginning, args):
    input_ids = tokenizer_gen.encode(beginning, return_tensors='pt')
    output = model_gen.generate(input_ids, **args)
    eos_n = [(output_i == tokenizer_gen.eos_token_id).nonzero() for output_i in
             output]
    return [tokenizer_gen.decode(
        output_i[:(eos_ni[0] if len(eos_ni) else len(output_i))],
        skip_special_tokens=False) for output_i, eos_ni in zip(output, eos_n)]


def generate_answer(question, args):
    input_ids = tokenizer_ans(question, return_tensors="pt").input_ids
    outputs = model_ans.generate(input_ids, **args)
    return tokenizer_ans.decode(outputs[0], skip_special_tokens=True)


@bot.message_handler(content_types=["text"])
def get_question(message):
    args = {'max_length': 70,
            'do_sample': True,
            'top_p': 0.7,
            'no_repeat_ngram_size': 3,
            'top_k': 0,
            }
    args_ans = {
        'do_sample': True,
        'temperature': 0.8,
    }
    if message.text in ['/help', '/start']:
        result = HELP_MSG
    else:
        splitted = message.text.split(' ')
        if splitted[0] in ['/answer', '/a']:
            if len(splitted) == 1:
                result = HELP_MSG
            else:
                result = generate_answer(message.text, args_ans)
        else:
            text = message.text
            if splitted[0] in ['/question', '/q']:
                if len(splitted) > 1:
                    text = ' '.join(text.split(' ')[1:])
                else:
                    text = ' '
            question = generate_text(text, args)
            answer = generate_answer(question, args_ans)
            result = f'ВОПРОС: {question[0]}\nОТВЕТ: {answer}'
    log_txt = f'{datetime.now()}: {message.from_user.username}: {message.text}:\n{result}'
    print(log_txt)
    try:
        log(log_txt)
    except:
        pass
    bot.send_message(message.chat.id, result)


def log(msg):
    with open(log_file, 'a') as f:
        f.write(msg)
        f.write('\n\n')


@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, "Hi! I generate CHGK questions")


bot.polling(none_stop=True, interval=0)
