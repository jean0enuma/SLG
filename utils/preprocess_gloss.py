import re
def process_text_llm(input_text):
    """
    process_textから結合処理を削除
    :param input_text:
    :return:
    """
    # 文字列の置換
    replacements = {
        r'loc-': '',
        r'cl-': '',
        r'qu-': '',
        r'poss-': '',
        r'lh-': '',
        r'S0NNE': 'SONNE',
        r'HABEN2': 'HABEN',
        r'__EMOTION__': '',
        r'__PU__': '',
        r'__LEFTHAND__': '',
        r'__EPENTHESIS__': '',
        r'WIE AUSSEHEN': 'WIE-AUSSEHEN',
        r'ZEIGEN ': 'ZEIGEN-BILDSCHIRM ',
        r'ZEIGEN$': 'ZEIGEN-BILDSCHIRM',
        r'-PLUSPLUS': ''
    }
    for pattern, replacement in replacements.items():
        input_text = re.sub(pattern, replacement, input_text)
    input_text = re.sub(r'([A-Z][A-Z])RAUM', r'\1', input_text)
    lines = input_text.split(' ')
    return remove_repetitions_single_sentence(lines)

def process_text(input_text):
    # 文字列の置換
    replacements = {
        r'loc-': '',
        r'cl-': '',
        r'qu-': '',
        r'poss-': '',
        r'lh-': '',
        r'S0NNE': 'SONNE',
        r'HABEN2': 'HABEN',
        r'__EMOTION__': '',
        r'__PU__': '',
        r'__LEFTHAND__': '',
        r'__EPENTHESIS__': '',
        r'WIE AUSSEHEN': 'WIE-AUSSEHEN',
        r'ZEIGEN ': 'ZEIGEN-BILDSCHIRM ',
        r'ZEIGEN$': 'ZEIGEN-BILDSCHIRM',
        r'-PLUSPLUS': ''
    }
    # ,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > tmp.ctm
    for pattern, replacement in replacements.items():
        input_text = re.sub(pattern, replacement, input_text)


    # 大文字の連続する単語の処理
    input_text = re.sub(r'\b([A-Z]+) \1\b', r'\1', input_text)
    # 特定のパターンの連結
    input_text = re.sub(r'^([A-Z]) ([A-Z][+ ])', r'\1+\2', input_text)
    input_text = re.sub(r'[ +]([A-Z]) ([A-Z]) ', r' \1+\2 ', input_text)
    input_text = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', input_text)
    input_text = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', input_text)
    input_text = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', input_text)

    input_text = re.sub(r'([ +]SCH) ([A-Z][ +])', r'\1+\2', input_text)
    input_text = re.sub(r'([ +]NN) ([A-Z][ +])', r'\1+\2', input_text)
    input_text = re.sub(r'([ +][A-Z]) (NN[ +])', r'\1+\2', input_text)
    input_text=re.sub(r'([ +][A-Z]) ([A-Z])$',r'\1+\2',input_text)
    input_text = re.sub(r'([A-Z][A-Z])RAUM', r'\1', input_text)
    #input_text = re.sub(r"__ON__", "<s>", input_text)
    #input_text=re.sub(r"__OFF__","</s>",input_text)
    # 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'を追加
    # input_text = re.sub(r'([ +][A-Z]) ([NN][ +])', r'\1+\2', input_text)
    # 特定パターンの行を除外
    lines = input_text.split(' ')

    return remove_repetitions_single_sentence(lines)


def process_hypothesis(text):
    line=text
    # Remove lines containing __LEFTHAND__, __EPENTHESIS__, or __EMOTION__
    line=re.sub(r'__LEFTHAND__|__EPENTHESIS__|__EMOTION__', '', line)

    # Remove words starting and ending with "__"
    line = re.sub(r'\b__[^_\s]*__\b', '', line)

    # Remove -PLUSPLUS suffixes
    line = re.sub(r'-PLUSPLUS\b', '', line)

    # Remove cl- and loc- prefixes
    line = re.sub(r'\b(cl-|loc-)(\S+)', r'\2', line)

    # Remove RAUM at the end of words
    line = re.sub(r'\b([A-Z][A-Z]*)RAUM\b', r'\1', line)

    # Remove trailing whitespace
    line = line.rstrip()


    # Remove repetitions (this is a simplified version and may not catch all cases)
    return remove_repetitions_single_sentence(line.split())


def remove_repetitions_single_sentence(words):
    # Initialize the result list with the first word (if it exists)
    while '' in words:
        words.remove('')
    result=words[:1]
    # Process each word starting from the second word
    for word in words[1:]:
        if word=='':
            continue
        # Add the word to the result if it's different from the previous word
        if word != result[-1]:
            result.append(word)

    # Join the words back into a sentence
    return ' '.join(result)


def process_reference(text):
    processed_lines = []
    line=text
    # Remove __LEFTHAND__, __EPENTHESIS__, and __EMOTION__
    line = re.sub(r'__LEFTHAND__|__EPENTHESIS__|__EMOTION__', '', line)

    # Remove words starting and ending with "__"
    line = re.sub(r'\b__[^_\s]*__\b', '', line)

    # Remove -PLUSPLUS suffixes
    line = re.sub(r'-PLUSPLUS\b', '', line)

    # Remove cl- and loc- prefixes
    line = re.sub(r'\b(cl-|loc-)(\S+)', r'\2', line)

    # Remove RAUM at the end of words
    line = re.sub(r'\b([A-Z][A-Z]*)RAUM\b', r'\1', line)

    # Convert "WIE AUSSEHEN" to "WIE-AUSSEHEN"
    line = line.replace('WIE AUSSEHEN', 'WIE-AUSSEHEN')

    # Add spelling letters to compounds
    # sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;' > tmp.stm
    #line=re.sub(r'^([A-Z]) ([A-Z][+ ])',r'\1+\2',line)
    ##line=re.sub(r'[ +]([A-Z]) ([A-Z])' , r'\1+\2',line)
    line=re.sub(r'([ +][A-Z]) ([A-Z][ +])',r'\1+\2',line)
    line=re.sub(r'([ +][A-Z]) ([A-Z][ +])',r'\1+\2',line)
    line=re.sub(r'([ +][A-Z]) ([A-Z][ +])',r'\1+\2',line)
    line=re.sub(r'([ +]SCH) ([A-Z][ +])',r'\1+\2',line)
    line=re.sub(r'([ +]NN) ([A-Z][ +])',r'\1+\2',line)
    line=re.sub(r'([ +][A-Z]) (NN[ +])',r'\1+\2',line)
    line=re.sub(r'([ +][A-Z]) ([A-Z])$',r'1+2',line)
    # 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g

    # Remove repetitions (this is a simplified version and may not catch all cases)
    return remove_repetitions_single_sentence(line.split())


if __name__ == "__main__":
    # 使用例
    input_text = 'A A'
    print(input_text)
    processed_text = process_text(input_text.upper())
    print(processed_text)
