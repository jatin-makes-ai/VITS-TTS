import phonemizer
from phonemizer.backend import EspeakBackend

# The set of symbols VITS usually uses (IPA + Punctuation)
_pad = "_"
_punctuation = " !',.:;? "
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɡɢɣɤhɦħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘppfqvʁɽɾɺɻʀʁʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"

# Final symbol list for the model's embedding layer
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

def text_to_phonemes(text, lang="en-us"):
    """Converts raw text to IPA phonemes using espeak-ng."""
    backend = EspeakBackend(lang, preserve_punctuation=True, with_stress=True)
    phonemes = backend.phonemize([text], strip=True)[0]
    return phonemes

if __name__ == "__main__":
    test_text = "Hello! I am building a low-latency voice model."
    print(f"Original: {test_text}")
    print(f"Phonemes: {text_to_phonemes(test_text)}")