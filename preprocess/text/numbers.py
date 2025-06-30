import re


_decimal_number_re = re.compile(r'([0-9]+,[0-9]+)')
_currency_re = re.compile(r'([€£$])([0-9]+(?:,[0-9]+)?)')
_ordinal_re = re.compile(r'([0-9]+)\.')
_number_re = re.compile(r'[0-9]+')


def normalize_numbers(text):
    text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_decimal_number_re, _expand_decimal_number, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def _expand_currency(match):
    symbol = match.group(1)
    number = match.group(2)
    number_word = _expand_number_text(number)
    currency_word = _get_currency_word(symbol, number)
    return f"{number_word} {currency_word}"


def _get_currency_word(symbol, number):
    number_value = float(number.replace(',', '.'))
    if symbol == '$':
        word = 'dolar' if number_value == 1 else 'dolara'
    elif symbol == '€':
        word = 'euro' if number_value == 1 else 'eura'
    elif symbol == '£':
        word = 'funta' if number_value == 1 else 'funti'
    else:
        word = ''
    return word


def _expand_decimal_number(match):
    number = match.group(0)
    integer_part, decimal_part = number.split(',')
    integer_word = _expand_number_text(integer_part)
    decimal_words = ' '.join([_digit_to_word(d) for d in decimal_part])
    cijelo_word = _get_cijelo_word(integer_part)
    return f"{integer_word} {cijelo_word} {decimal_words}"


def _get_cijelo_word(integer_part):
    last_digit = int(integer_part[-1])
    if integer_part == '1' or integer_part == '0':
        return 'cijelo'
    elif last_digit == 1 and not integer_part.endswith('11'):
        return 'cijelo'
    elif last_digit in [2,3,4] and not integer_part.endswith(('12','13','14')):
        return 'cijela'
    else:
        return 'cijelih'


def _expand_ordinal(match):
    number = match.group(1)
    following_text = match.string[match.end():]
    next_word_match = re.match(r'\s*(\S+)', following_text)
    if next_word_match:
        next_word = next_word_match.group(1)
        gender = _get_gender(next_word)
    else:
        gender = 'm'  
    ordinal_word = _number_to_ordinal(int(number), gender)
    return ordinal_word


def _get_gender(word):
    if word.endswith('a'):
        return 'f' 
    elif word[-1] in 'aeiou':
        return 'n' 
    else:
        return 'm' 


def _expand_number(match):
    number = match.group(0)
    return _expand_number_text(number)


def _expand_number_text(number):
    num = int(number)
    return _number_to_words(num)


def _number_to_words(num):
    units = ['', 'jedan', 'dva', 'tri', 'četiri', 'pet', 'šest', 'sedam', 'osam', 'devet']
    teens = ['deset', 'jedanaest', 'dvanaest', 'trinaest', 'četrnaest', 'petnaest',
             'šesnaest', 'sedamnaest', 'osamnaest', 'devetnaest']
    tens = ['', '', 'dvadeset', 'trideset', 'četrdeset', 'pedeset',
            'šezdeset', 'sedamdeset', 'osamdeset', 'devedeset']
    hundreds = ['', 'sto', 'dvjesto', 'tristo', 'četiristo',
                'petsto', 'šeststo', 'sedamsto', 'osamsto', 'devetsto']

    if num == 0:
        return 'nula'
    elif num < 10:
        return units[num]
    elif num < 20:
        return teens[num - 10]
    elif num < 100:
        ten = tens[num // 10]
        unit = units[num % 10]
        return ten + ('' if unit == '' else ' ' + unit)
    elif num < 1000:
        hundred = hundreds[num // 100]
        remainder = num % 100
        if remainder:
            return hundred + ' ' + _number_to_words(remainder)
        else:
            return hundred
    elif num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        if thousands == 1:
            thousands_word = 'tisuću'
        else:
            thousands_word = _number_to_words(thousands) + ' tisuće' if 2 <= thousands <= 4 else _number_to_words(thousands) + ' tisuća'
        if remainder:
            return thousands_word + ' ' + _number_to_words(remainder)
        else:
            return thousands_word
    elif num < 1000000000:
        millions = num // 1000000
        remainder = num % 1000000
        if millions == 1:
            millions_word = 'milijun'
        else:
            millions_word = _number_to_words(millions) + ' milijuna'
        if remainder:
            return millions_word + ' ' + _number_to_words(remainder)
        else:
            return millions_word
    elif num < 1000000000000:
        billions = num // 1000000000
        remainder = num % 1000000000
        if billions == 1:
            billions_word = 'milijarda'
        else:
            billions_word = _number_to_words(billions) + ' milijarde'
        if remainder:
            return billions_word + ' ' + _number_to_words(remainder)
        else:
            return billions_word
    else:
        return str(num)


def _digit_to_word(digit):
    digits = ['nula', 'jedan', 'dva', 'tri', 'četiri', 'pet', 'šest', 'sedam', 'osam', 'devet']
    return digits[int(digit)]


def _number_to_ordinal(num, gender):
    ordinals_masc = ['nulti', 'prvi', 'drugi', 'treći', 'četvrti', 'peti', 'šesti',
                     'sedmi', 'osmi', 'deveti', 'deseti']
    ordinals_fem = ['nulta', 'prva', 'druga', 'treća', 'četvrta', 'peta', 'šesta',
                    'sedma', 'osma', 'deveta', 'deseta']
    ordinals_neu = ['nulto', 'prvo', 'drugo', 'treće', 'četvrto', 'peto', 'šesto',
                    'sedmo', 'osmo', 'deveto', 'deseto']
    if num < 11:
        if gender == 'm':
            return ordinals_masc[num]
        elif gender == 'f':
            return ordinals_fem[num]
        else:
            return ordinals_neu[num]
    else:
        base = _number_to_words(num)
        if gender == 'm':
            suffix = 'i'
        elif gender == 'f':
            suffix = 'a'
        else:
            suffix = 'o'
        return base + suffix