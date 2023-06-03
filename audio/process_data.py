import nltk
from nltk.tokenize import word_tokenize
from sinling import SinhalaTokenizer, SinhalaStemmer as stemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from nltk.tokenize.treebank import TreebankWordDetokenizer
from audio.translate_text import translate_en_to_si
from audio.singlish import convert_singlish_to_sinhala_text
from lib.SinhaleseVowelLetterFixer import SinhaleseVowelLetterFixer
from lib.StemWords import stem_word

nltk.download('punkt')
stemmer = stemmer()
tokenizer = SinhalaTokenizer()


def stem_word(word: str) -> str:
    # word= translate_to_sinhala(word)
    """
    Stemming words
    :param word: word
    :return: stemmed word
    """
    if len(word) < 4:
        return word

    # remove 'ට'
    if word[-1] == 'ට':
        return word[:-1]

    # remove 'ම'
    if word[-1] == 'ම':
        return word[:-1]

    # remove 'ද'
    if word[-1] == 'ද':
        return word[:-1]

    # remove 'ටත්'
    if word[-3:] == 'ටත්':
        return word[:-3]

    # remove 'එක්'
    if word[-3:] == 'ෙක්':
        return word[:-3]

    # remove 'යේ'
    if word[-2:] == 'යේ':
        return word[:-2]

    # remove 'ගෙ' (instead of ගේ because this step comes after simplifying text)
    if word[-2:] == 'ගෙ':
        return word[:-2]

    # remove 'එ'
    if word[-1:] == 'ෙ':
        return word[:-1]

    # remove 'ක්'
    if word[-2:] == 'ක්':
        return word[:-2]

    # remove 'වත්'
    if word[-3:] == 'වත්':
        return word[:-3]

    word = stemmer.stem(word)
    word = word[0]

    # else
    return word


def filter_stop_words(sentences):
    stopwords_set = ["සහ", "සමග", "සමඟ", "අහා", "ආහ්", "ආ", "ඕහෝ", "අනේ", "අඳෝ", "අපොයි", "පෝ", "අයියෝ", "ආයි", "ඌයි",
                     "චී", "චිහ්", "චික්", "හෝ‍", "දෝ",
                     "දෝහෝ", "මෙන්", "සේ", "වැනි", "බඳු", "වන්", "අයුරු", "අයුරින්", "ලෙස", "වැඩි", "ශ්‍රී", "හා", "ය",
                     "නිසා", "නිසාවෙන්", "බවට", "බව", "බවෙන්", "නම්", "වැඩි", "සිට",
                     "දී", "මහා", "මහ", "පමණ", "පමණින්", "පමන", "වන", "විට", "විටින්", "මේ", "මෙලෙස", "මෙයින්", "ඇති",
                     "ලෙස", "සිදු", "වශයෙන්", "යන", "සඳහා", "මගින්", "හෝ‍",
                     "ඉතා", "ඒ", "එම", "ද", "අතර", "විසින්", "සමග", "පිළිබඳව", "පිළිබඳ", "තුළ", "බව", "වැනි", "මහ",
                     "මෙම", "මෙහි", "මේ", "වෙත", "වෙතින්", "වෙතට", "වෙනුවෙන්",
                     "වෙනුවට", "වෙන", "ගැන", "නෑ", "අනුව", "නව", "පිළිබඳ", "විශේෂ", "දැනට", "එහෙන්", "මෙහෙන්", "එහේ",
                     "මෙහේ", "ම", "තවත්", "තව", "සහ", "දක්වා", "ට", "ගේ",
                     "එ", "ක", "ක්", "බවත්", "බවද", "මත", "ඇතුලු", "ඇතුළු", "මෙසේ", "වඩා", "වඩාත්ම", "නිති", "නිතිත්",
                     "නිතොර", "නිතර", "ඉක්බිති", "දැන්", "යලි", "පුන", "ඉතින්",
                     "සිට", "සිටන්", "පටන්", "තෙක්", "දක්වා", "සා", "තාක්", "තුවක්", "පවා", "ද", "හෝ‍", "වත්", "විනා",
                     "හැර", "මිස", "මුත්", "කිම", "කිම්", "ඇයි", "මන්ද", "හෙවත්",
                     "නොහොත්", "පතා", "පාසා", "ගානෙ", "තව", "ඉතා", "බොහෝ", "වහා", "සෙද", "සැනින්", "හනික", "එම්බා",
                     "එම්බල", "බොල", "නම්", "වනාහි", "කලී", "ඉඳුරා",
                     "අන්න", "ඔන්න", "මෙන්න", "උදෙසා", "පිණිස", "සඳහා", "රබයා", "නිසා", "එනිසා", "එබැවින්", "බැවින්",
                     "හෙයින්", "සේක්", "සේක", "ගැන", "අනුව", "පරිදි", "විට",
                     "තෙක්", "මෙතෙක්", "මේතාක්", "තුරු", "තුරා", "තුරාවට", "තුලින්", "නමුත්", "එනමුත්", "වස්", 'මෙන්',
                     "ලෙස", "පරිදි", "එහෙත්"]

    filtered_sentences = []
    detokenizer = Detok()
    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        filtered_sentence = [word for word in tokenized_sentence if word not in stopwords_set]
        filtered_sentence = []
        for w in tokenized_sentence:
            if w not in stopwords_set:
                filtered_sentence.append(stem_word(w))
        filtered_sentences.append(filtered_sentence)

    # remove empty lists
    filtered_sentences = [x for x in filtered_sentences if x != []]

    return filtered_sentences


def Detokenioze(text):
    detokenized_sentences = []

    for sentence in text:
        detokenized_sentences.append(TreebankWordDetokenizer().detokenize(sentence))
    return detokenized_sentences


def clean_data(text: str):
    # replace URL of a text
    text = text.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
    # replace mention
    text = text.replace('#|@\w*', '')
    # remove retweet states in the beginning such as "RT @sam92ky: "
    text = text.replace('RT : ', '')
    # remove numbers
    text = text.replace('\d+', '')

    # punctuation removal
    text = text.replace('[^\w\s]', '')

    # remove emoji
    text = text.replace('[^\w\s#@/:%.,_-]', '')
    # remove extra white spaces
    text = text.replace('\s+', ' ')
    # remove white space at the beginning
    text = text.replace('^\s', '')
    # remove white space at the end
    text = text.replace('\s$', '')

    return text


######## Simplify sinhalese text ########
# dictionary that maps wrong usage of vowels to correct vowels
simplify_characters_dict = {
    # Consonant
    "ඛ": "ක",
    "ඝ": "ග",
    "ඟ": "ග",
    "ඡ": "ච",
    "ඣ": "ජ",
    "ඦ": "ජ",
    "ඤ": "ඥ",
    "ඨ": "ට",
    "ඪ": "ඩ",
    "ණ": "න",
    "ඳ": "ද",
    "ඵ": "ප",
    "භ": "බ",
    "ඹ": "බ",
    "ශ": "ෂ",
    "ළ": "ල",

    # Vowels
    "ආ": "අ",
    "ඈ": "ඇ",
    "ඊ": "ඉ",
    "ඌ": "උ",
    "ඒ": "එ",
    "ඕ": "ඔ",

}


def get_simplified_character(character: str) -> str:
    if len(character) != 1:
        raise TypeError("character should be a string with length 1")
    try:
        return simplify_characters_dict[character]
    except KeyError:
        return character


def simplify_sinhalese_text(text: str) -> str:
    """
    simplify
    :param text:
    :return:
    """
    modified_text = ""
    for c in text:
        modified_text += get_simplified_character(c)
    return modified_text


def process_sentence(text):
    print("Original text:", text, '\n')

    # clean data
    text = clean_data(text)
    print("Cleaned text:", text, '\n')

    # translate
    text = translate_en_to_si(text)
    print("Translated text:", text, '\n')

    # convert singlish to Sinhala
    text = convert_singlish_to_sinhala_text(text)
    print("converted text:", text, '\n')

    # split text into sentences
    sentences = tokenizer.tokenize(text)
    print("Tokenized sentences: ", sentences, '\n')

    # remove stopwords
    filtered_sentences = filter_stop_words(sentences)
    print("Filtered sentences: ", filtered_sentences, '\n')

    # detokenize
    detokenized_sentences = Detokenioze(filtered_sentences)
    print("Detokenized sentences: ", detokenized_sentences, '\n')

    # simplify
    simplified_sentences = []
    for sentence in detokenized_sentences:
        simplified_sentences.append(simplify_sinhalese_text(sentence))
    print("Simplified sentences: ", simplified_sentences, '\n')

    # vowel letter fixer
    fixed_sentences = []
    for sentence in simplified_sentences:
        fixed_sentences.append(SinhaleseVowelLetterFixer.get_fixed_text(sentence))
    print("Fixed sentences: ", fixed_sentences, '\n')

    # join sentences
    joined_sentences = ' '.join(fixed_sentences)
    print("Joined sentences: ", joined_sentences, '\n')

    return joined_sentences


def process_data_df(df, column_name='comment'):
    # get unique values in column_name
    unique_values = df[column_name].unique()
    print("Unique values: ", unique_values, '\n')

    # process each unique value call process_data()
    for value in unique_values:
        # call process_data()
        processed_data = process_sentence(value)
        print("Processed data: ", processed_data, '\n')
        print("Processe df: ", df, '\n')

        # if column_name = value in df
        if value in df[column_name].values:
            # add processed_text new column with processed_data
            df.loc[df[column_name] == value, 'processed_text'] = processed_data
            print("df: ", df, '\n')

    return df
