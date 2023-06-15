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
        # print(char)
        if char in vowel_mapping:
            if i + 1 < len(text) and text[i:i + 2] in depended_vowl_mapping:
                converted_text += depended_vowl_mapping[text[i:i + 2]]
                i += 2
            else:
                converted_text += vowel_mapping[char]
                i += 1
        elif char in constant_mapping:
            if i + 1 < len(text) and text[i:i + 2] in constant_mapping:
                converted_text += constant_mapping[text[i:i + 2]]
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

    try:
        # print(le)
        for i in range(len(le)):
            if le[i] == '්' and i + 1 < len(le) and le[i + 1] in depended_vowl_mapping.values():
                le[i] = ''

        # # remove empty elements
        # le_a = [x for x in le if x != '']
        # print(le_a)
    except Exception as e:
        print(e)
        pass

    return join_to_word(le)


def fix_first_letter_vowel(word):
    le = separate_to_letters(word)

    # print(le)

    try:
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
    except Exception as e:
        print(e)
        pass

    return join_to_word(le)

# def fix_last_letter_vowel(word):
#     le = separate_to_letters(word)
#
#     print(le)
#
#     try:
#         # if first letter is depended_vowl_mapping , replace with coreesponding vowel_mapping
#         if le[-1] in vowel_only_mapping.values() and le[-2] in vowel_only_mapping.values() and le[-3] in constant_mapping.values():
#             # find le[0] in depended_vowl_mapping.values()
#             for key, value in vowel_only_mapping.items():
#                 if value == le[-1]:
#                     en_letter = key
#                     # print(en_letter)
#                     break
#
#             # print(vowel_only_mapping[en_letter])
#
#             le[-1] = depended_vowl_mapping[en_letter]
#
#     except Exception as e:
#         print(e)
#         pass
#
#     return join_to_word(le)

def fix_const_plus_a(word):
    le = separate_to_letters(word)
    # print(le)

    # if '්', 'අ' found in any place, then remove '්' and 'අ'
    try:
        for i in range(len(le)):
            if le[i] == '්' and i + 1 < len(le) and le[i + 1] == 'අ':
                le[i] = ''
                le[i + 1] = ''
    except Exception as e:
        print(e)
        pass

    # print(le)

    return join_to_word(le)


def convert_singlish_to_sinhala_text(input_text):
    output_text = ""

    for word in input_text.split():
        output_text += fix_const_plus_a(fix_first_letter_vowel(fix_word(convert_to_sinhala(word)))) + " "
        # output_text += fix_last_letter_vowel(fix_first_letter_vowel(fix_word(convert_to_sinhala(word)))) + " "

    # print(output_text)
    return output_text


# print(convert_singlish_to_sinhala_text("a"))
# print(convert_singlish_to_sinhala_text("aa"))
# print(convert_singlish_to_sinhala_text("aaa"))
# print(convert_singlish_to_sinhala_text("aayuboewan"))
# print(convert_singlish_to_sinhala_text("naa"))
# print(convert_singlish_to_sinhala_text("nae nai daayoe"))