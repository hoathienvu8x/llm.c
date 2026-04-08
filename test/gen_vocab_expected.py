#!/usr/bin/env python3
"""Generate expected tokenization outputs for vocab tests.

Uses tiktoken with merge rules from the GGUF file to produce
reference token IDs for each test string. Only works for BPE
models (tokenizer.ggml.model == "gpt2"). For SentencePiece models,
we use a greedy longest-match approach since sentencepiece isn't
available.

Usage: python3 test/gen_vocab_expected.py <model.gguf> > test/expected_vocab_<name>.txt
"""

import struct, sys, regex, os

def load_gguf_vocab(path):
    tokens, merges, tok_model = [], [], None
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def read_string():
            slen = struct.unpack('<Q', f.read(8))[0]
            if slen > 10_000_000:
                f.seek(slen, 1)
                return ''
            return f.read(slen).decode('utf-8', errors='replace')

        def skip_value(vtype):
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:4, 11:8, 12:8, 13:8}
            if vtype == 8:
                slen = struct.unpack('<Q', f.read(8))[0]
                f.seek(slen, 1)
            elif vtype == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for _ in range(arr_len):
                    skip_value(arr_type)
            elif vtype in sizes:
                f.seek(sizes[vtype], 1)

        for _ in range(n_kv):
            key = read_string()
            vtype = struct.unpack('<I', f.read(4))[0]
            if key == 'tokenizer.ggml.tokens':
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for i in range(arr_len):
                    tokens.append(read_string())
            elif key == 'tokenizer.ggml.merges':
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for i in range(arr_len):
                    merges.append(read_string())
            elif key == 'tokenizer.ggml.model':
                tok_model = read_string()
            else:
                skip_value(vtype)

    return tokens, merges, tok_model


def make_bpe_tokenizer(tokens_list, merges_list):
    """Build a tiktoken Encoding from GGUF vocab and merges."""
    import tiktoken

    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    mergeable_ranks = {}
    for i, tok in enumerate(tokens_list):
        if i >= 128000:
            continue
        try:
            raw = bytes([byte_decoder[c] for c in tok])
            mergeable_ranks[raw] = i
        except KeyError:
            pass

    special_tokens = {tok: i for i, tok in enumerate(tokens_list) if i >= 128000}

    pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    return tiktoken.Encoding(
        name="llama3",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def make_spm_tokenizer(tokens_list):
    """Build a greedy longest-match tokenizer for SentencePiece models."""
    decoded = {}
    for i, tok in enumerate(tokens_list):
        # spm_decode: replace ▁ with space, decode <0xNN> escapes
        d = tok.replace('\u2581', ' ')
        # Handle <0xNN> byte escapes
        import re
        while True:
            m = re.search(r'<0x([0-9A-Fa-f]{2})>', d)
            if not m:
                break
            d = d[:m.start()] + chr(int(m.group(1), 16)) + d[m.end():]
        if d not in decoded:
            decoded[d] = i  # first (lowest index) wins

    max_len = max(len(k) for k in decoded)

    def tokenize(text):
        tokens = []
        pos = 0
        while pos < len(text):
            # Special token check: <|...|>
            if text[pos:pos+2] == '<|':
                end = text.find('|>', pos + 2)
                if end >= 0:
                    slen = end + 2 - pos
                    candidate = text[pos:pos+slen]
                    if candidate in decoded:
                        tokens.append(decoded[candidate])
                        pos += slen
                        continue
            # Special token check: [...]
            if text[pos] == '[':
                end = text.find(']', pos + 1)
                if end >= 0:
                    slen = end + 1 - pos
                    candidate = text[pos:pos+slen]
                    if candidate in decoded:
                        tokens.append(decoded[candidate])
                        pos += slen
                        continue

            # Don't let greedy match cross into special token boundaries
            remaining = len(text) - pos
            scan = pos + 1
            while scan < len(text):
                if text[scan:scan+2] == '<|':
                    remaining = scan - pos
                    break
                if text[scan] == '[':
                    end2 = text.find(']', scan + 1)
                    if end2 >= 0 and text[scan:end2+1] in decoded:
                        remaining = scan - pos
                        break
                scan += 1

            # Greedy longest match
            best_len, best_idx = 0, -1
            limit = min(max_len, remaining)
            for l in range(1, limit + 1):
                candidate = text[pos:pos+l]
                if candidate in decoded:
                    best_len = l
                    best_idx = decoded[candidate]
            if best_idx >= 0:
                tokens.append(best_idx)
                pos += best_len
            else:
                pos += 1  # skip unknown byte
        return tokens

    return tokenize


# Test strings that exercise various tokenization edge cases
TEST_STRINGS = [
    # Simple text
    "hello world",
    # Contractions
    "what's the capital of france?",
    # Numbers (critical: 3-digit grouping)
    "In 2023, there were 1000 people",
    # System preamble text
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
    # Special tokens
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|>",
    # Mixed with punctuation
    "Hello! How are you? I'm fine.",
    # Full Llama 3 style prompt (no BOS)
    "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    # Mistral style prompt
    "[INST] What is the capital of France? [/INST]",
    # Whitespace edge cases
    "  hello  ",
    "\n\n\n",
    # Numbers edge cases
    "12345678",
    "2024-01-15",
]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.gguf>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    tokens_list, merges_list, tok_model = load_gguf_vocab(path)
    print(f"# model: {os.path.basename(path)}", file=sys.stderr)
    print(f"# tokenizer: {tok_model}, {len(tokens_list)} tokens, {len(merges_list)} merges", file=sys.stderr)

    is_bpe = tok_model != "llama"

    if is_bpe and merges_list:
        import tiktoken
        enc = make_bpe_tokenizer(tokens_list, merges_list)
        def tokenize(text):
            return list(enc.encode(text, allowed_special="all"))
    else:
        tokenize = make_spm_tokenizer(tokens_list)

    # Output format: one test per line
    # TEXT\tTOK1 TOK2 TOK3 ...
    for text in TEST_STRINGS:
        toks = tokenize(text)
        escaped = text.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t')
        tok_str = ' '.join(str(t) for t in toks)
        print(f"{escaped}\t{tok_str}")


if __name__ == '__main__':
    main()
