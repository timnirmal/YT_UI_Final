import re

from deep_translator import GoogleTranslator


def translate_en_to_si(text):
    tr = GoogleTranslator(source='en', target='si')

    english_words = []
    english_words_list = []
    # get adjacent english words together
    for word in text.split():
        # check if word is english
        if word.isalpha():
            english_words.append(word)
        else:
            if len(english_words) > 0:
                english_words_list.append(' '.join(english_words))
                english_words = []
            english_words_list.append(word)

    # translate english words to sinhala
    for i in range(len(english_words_list)):
        # if word is english, space, or punctuation
        pattern = r'[a-zA-Z\s]'
        if re.match(pattern, english_words_list[i]):
            english_words_list[i] = tr.translate(english_words_list[i])

    # join the words back together
    text = ' '.join(english_words_list)

    return tr.translate(text)


txt = "hi how are you අසාර්ථකයි අසාර්ථක වීමට mama nm hodain innawa මේ ගැන ඔබතුමාට කියන්න තියෙන්නෙ"
# print(translate_en_to_si(txt))
