import re
import json

print("Reading catalog.json...")
with open('catalog.json', 'rb') as f:
    raw = f.read()

print("Cleaning...")
text = raw.decode('utf-8', errors='ignore')

# This regex finds control characters INSIDE JSON strings and removes them
# It replaces newlines/tabs inside quoted strings with a space
def clean_json_strings(text):
    result = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            result.append(char)
            escaped = False
        elif char == '\\' and in_string:
            result.append(char)
            escaped = True
        elif char == '"':
            in_string = not in_string
            result.append(char)
        elif in_string and char in ('\n', '\r', '\t', '\x00'):
            result.append(' ')  # replace bad char with space
        elif in_string and ord(char) < 32:
            result.append(' ')  # replace any other control char
        else:
            result.append(char)
    return ''.join(result)

print("Fixing control characters inside strings...")
text = clean_json_strings(text)

print("Parsing JSON...")
try:
    catalog = json.loads(text)
    print(f"Success! {len(catalog)} assessments found.")
    
    with open('catalog.json', 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print("catalog.json cleaned and saved.")

except json.JSONDecodeError as e:
    print(f"Still failing at: {e}")
    pos = e.pos
    print(f"Context around error:")
    print(repr(text[max(0,pos-150):pos+150]))