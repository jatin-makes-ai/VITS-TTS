# utils/text_utils.py
_pad = "_"
_punctuation = " !',.:;? "
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɡɢɣɤhɦħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘppfqvʁɽɾɺɻʀʁʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"

# The master list of symbols
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Mapping: character -> index
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text):
    """Converts a string of phonemes into a list of integers."""
    sequence = []
    for symbol in text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
    return sequence