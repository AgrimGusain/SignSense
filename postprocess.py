from collections import deque
import difflib

# ✅ Buffer to hold incoming letters
letter_buffer = deque()

# ✅ Predefined vocabulary — ONLY these can be output
VOCAB = ["HELLO", "HEY", "WAY", "KEY", "BAY", "PAY"]

# ✅ Add new letters to buffer
def add_detected_letters(detected_letters):
    letter_buffer.extend(detected_letters)

# ✅ Clean repeated rapid detections like ['H','H','H','E','E'] → ['H','E']
def clean_letters():
    cleaned = []
    prev = None
    count = 0

    for l in letter_buffer:
        if l == prev:
            count += 1
        else:
            if prev is not None and count >= 2:
                cleaned.append(prev)
            prev = l
            count = 1

    if prev is not None and count >= 2:
        cleaned.append(prev)

    return cleaned

# ✅ Only choose closest match from vocabulary
def correct_with_nlp(letter_list):
    input_word = ''.join(letter_list).upper()
    closest = difflib.get_close_matches(input_word, VOCAB, n=1, cutoff=0.6)

    if closest:
        return closest[0]  # ✅ Only return match if found
    else:
        return ""  # ❌ No good match — return empty
       
# ✅ Clear buffer after each word
def clear_buffer():
    letter_buffer.clear()
