vowel_only_mapping = {
    'a': 'අ',
    'aa': 'ආ',
    'ae': 'ඇ',
    'aae': 'ඈ',
    'i': 'ඉ',
    'ii': 'ඊ',
    'u': 'උ',
    'uu': 'ඌ',
    'e': 'එ',
    'ee': 'ඒ',
    'ai': 'ඓ',
    'o': 'ඔ',
    'oo': 'ඕ',
    'au': 'ඖ'
}

vowel_mapping = {
    'a': 'අ',
    'aa': 'ආ',
    'ae': 'ඇ',
    'aae': 'ඈ',
    'i': 'ඉ',
    'ii': 'ඊ',
    'u': 'උ',
    'uu': 'ඌ',
    'e': 'එ',
    'ee': 'ඒ',
    'ai': 'ඓ',
    'o': 'ඔ',
    'oo': 'ඕ',
    'au': 'ඖ',
    'aa': 'ා',
    'ae': 'ැ',
    'aae': 'ෑ',
    'i': 'ි',
    'ii': 'ී',
    'u': 'ු',
    'uu': 'ූ',
    'e': 'ෙ',
    'ee': 'ේ',
    'ai': 'ෛ',
    'o': 'ො',
    'oe': 'ෝ',
    'au': 'ෞ',
}

constant_mapping = {
    'k': 'ක්',
    'kh': 'ඛ්',
    'g': 'ග්',
    'gh': 'ඝ්',
    'ng': 'ඞ්',
    'nng': 'ඟ්',
    'c': 'ච්',
    'ch': 'ඡ්',
    'j': 'ජ්',
    'jh': 'ඣ්',
    'ny': 'ඤ්',
    'jny': 'ඥ්',
    'tt': 'ට්',
    'tth': 'ඨ්',
    'dd': 'ඩ්',
    'ddh': 'ඪ්',
    'nn': 'ණ්',
    'nndd': 'ඬ්',
    't': 'ත්',
    'th': 'ථ්',
    'd': 'ද්',
    'dh': 'ධ්',
    'n': 'න්',
    'nd': 'ඳ්',
    'p': 'ප්',
    'ph': 'ඵ්',
    'b': 'බ්',
    'bh': 'භ්',
    'm': 'ම්',
    'mb': 'ඹ්',
    'y': 'ය්',
    'r': 'ර්',
    'l': 'ල්',
    'v': 'ව්',
    'w': 'ව්',
    'sh': 'ශ්',
    'ss': 'ෂ්',
    's': 'ස්',
    'h': 'හ්',
    'll': 'ළ්',
    'f': 'ෆ්',
}

constant_a_mapping = {
    'ka': 'ක',
    'kha': 'ඛ',
    'ga': 'ග',
    'gha': 'ඝ',
    'nga': 'ඞ',
    'nnga': 'ඟ',
    'ca': 'ච',
    'cha': 'ඡ',
    'ja': 'ජ',
    'jha': 'ඣ',
    'nya': 'ඤ',
    'jnya': 'ඥ',
    'tta': 'ට',
    'ttha': 'ඨ',
    'dda': 'ඩ',
    'ddha': 'ඪ',
    'nna': 'ණ',
    'nndda': 'ඬ',
    'ta': 'ත',
    'tha': 'ථ',
    'da': 'ද',
    'dha': 'ධ',
    'na': 'න',
    'ndda': 'ඳ',
    'pa': 'ප',
    'pha': 'ඵ',
    'ba': 'බ',
    'bha': 'භ',
    'ma': 'ම',
    'mba': 'ඹ',
    'ya': 'ය',
    'ra': 'ර',
    'la': 'ල',
    'va': 'ව',
    'wa': 'ව',
    'sha': 'ශ',
    'ssa': 'ෂ',
    'sa': 'ස',
    'ha': 'හ',
    'lla': 'ළ',
    'fa': 'ෆ',
}

depended_vowl_mapping = {
    'aa': 'ා',
    'ae': 'ැ',
    'aae': 'ෑ',
    'i': 'ි',
    'ii': 'ී',
    'u': 'ු',
    'uu': 'ූ',
    'e': 'ෙ',
    'ee': 'ේ',
    'ai': 'ෛ',
    'o': 'ො',
    'oe': 'ෝ',
    'au': 'ෞ',
}


def convert_to_sinhala(text):
    converted_text = ''
    i = 0
    while i < len(text):
        char = text[i]
        if char in vowel_mapping:
            if i + 1 < len(text) and text[i:i + 2] in depended_vowl_mapping:
                converted_text += depended_vowl_mapping[text[i:i + 2]]
                i += 2
            else:
                converted_text += vowel_mapping[char]
                i += 1
        elif char in constant_mapping:
            if i + 1 < len(text) and text[i:i + 2] in constant_a_mapping:
                converted_text += constant_a_mapping[text[i:i + 2]]
                i += 2
            else:
                converted_text += constant_mapping[char]
                i += 1
        else:
            converted_text += char
            i += 1
    return converted_text


def separate_to_letters(word):
    return list(word)


def join_to_word(letters):
    return ''.join(letters)


def fix_word(word):
    le = separate_to_letters(word)

    # print(le)
    for i in range(len(le)):
        if le[i] == '්' and i + 1 < len(le) and le[i + 1] in depended_vowl_mapping.values():
            le[i] = ''

    # # remove empty elements
    # le_a = [x for x in le if x != '']
    # print(le_a)

    return join_to_word(le)


def fix_first_letter_vowel(word):
    le = separate_to_letters(word)

    # print(le)

    # if first letter is depended_vowl_mapping , replace with coreesponding vowel_mapping
    if le[0] in depended_vowl_mapping.values():
        # find le[0] in depended_vowl_mapping.values()
        for key, value in depended_vowl_mapping.items():
            if value == le[0]:
                en_letter = key
                # print(en_letter)
                break

        # print(vowel_only_mapping[en_letter])

        le[0] = vowel_only_mapping[en_letter]

    return join_to_word(le)


def convert_singlish_to_sinhala_text(input_text):
    output_text = ""

    for word in input_text.split():
        output_text += fix_first_letter_vowel(fix_word(convert_to_sinhala(word))) + " "

    # print(output_text)
    return output_text
