#include "simple_tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <zlib.h>

// ============================================================================
// Utility: read gzip file
// ============================================================================

std::string SimpleTokenizer::read_gzip_file(const std::string& path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) {
        throw std::runtime_error("Cannot open gzip file: " + path);
    }
    std::string result;
    char buf[8192];
    int n;
    while ((n = gzread(gz, buf, sizeof(buf))) > 0) {
        result.append(buf, n);
    }
    gzclose(gz);
    return result;
}

std::vector<std::string> SimpleTokenizer::split_string(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string part;
    while (std::getline(iss, part, delim)) {
        parts.push_back(part);
    }
    return parts;
}

// ============================================================================
// bytes_to_unicode — exact port of Python function
// ============================================================================

static std::string codepoint_to_utf8(uint32_t cp) {
    // Convert a Unicode codepoint to its UTF-8 encoding
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

std::unordered_map<uint8_t, std::string> SimpleTokenizer::build_bytes_to_unicode() {
    // Direct-mapped byte ranges (printable, non-problematic):
    //   33..126   ('!' to '~')
    //   161..172  ('¡' to '¬')
    //   174..255  ('®' to 'ÿ')
    std::vector<uint8_t> bs;
    for (int i = 33; i <= 126; ++i)  bs.push_back(static_cast<uint8_t>(i));
    for (int i = 161; i <= 172; ++i) bs.push_back(static_cast<uint8_t>(i));
    for (int i = 174; i <= 255; ++i) bs.push_back(static_cast<uint8_t>(i));

    // cs starts as copy of bs (codepoints that map to themselves)
    std::vector<uint32_t> cs(bs.begin(), bs.end());

    // Remaining bytes (0..32, 127..160, 173) get mapped to codepoints 256+
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) == bs.end()) {
            bs.push_back(static_cast<uint8_t>(b));
            cs.push_back(256 + n);
            ++n;
        }
    }
    // n should be 68 (256 - 188 direct mappings)

    std::unordered_map<uint8_t, std::string> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result[bs[i]] = codepoint_to_utf8(cs[i]);
    }
    return result;
}

// ============================================================================
// Constructor — load BPE vocab
// ============================================================================

SimpleTokenizer::SimpleTokenizer(const std::string& bpe_path, int context_length)
    : context_length_(context_length) {
    // 1. Build byte encoder/decoder
    byte_encoder_ = build_bytes_to_unicode();
    for (auto& [k, v] : byte_encoder_) {
        byte_decoder_[v] = k;
    }

    // 2. Read gzip BPE vocab
    std::string content = read_gzip_file(bpe_path);
    std::vector<std::string> lines = split_string(content, '\n');

    // 3. Extract merges: skip header (line 0), take lines [1..48895)
    //    = 48894 merges
    const int num_merges = 49152 - 256 - 2;  // = 48894
    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(num_merges);
    for (int i = 1; i <= num_merges && i < static_cast<int>(lines.size()); ++i) {
        auto parts = split_string(lines[i], ' ');
        if (parts.size() >= 2) {
            merges.push_back({parts[0], parts[1]});
        }
    }

    // 4. Build vocabulary
    //    Phase 1: 256 base byte-unicode chars
    std::vector<std::string> vocab;
    vocab.reserve(49408);
    auto b2u = build_bytes_to_unicode();
    // Collect in byte order 0..255
    std::vector<std::string> base_chars(256);
    for (auto& [byte_val, unicode_str] : b2u) {
        base_chars[byte_val] = unicode_str;
    }
    for (auto& s : base_chars) {
        vocab.push_back(s);
    }

    //    Phase 2: 256 word-end variants (base + "</w>")
    for (auto& s : base_chars) {
        vocab.push_back(s + "</w>");
    }

    //    Phase 3: 48894 BPE merge tokens
    for (auto& [a, b] : merges) {
        vocab.push_back(a + b);
    }

    //    Phase 4: special tokens
    vocab.push_back("<start_of_text>");
    vocab.push_back("<end_of_text>");

    // 5. Build encoder/decoder
    for (int i = 0; i < static_cast<int>(vocab.size()); ++i) {
        encoder_[vocab[i]] = i;
        decoder_[i] = vocab[i];
    }

    // 6. Build BPE rank lookup
    for (int i = 0; i < static_cast<int>(merges.size()); ++i) {
        bpe_ranks_[merges[i]] = i;
    }

    // 7. Special token IDs
    sot_token_id_ = encoder_.at("<start_of_text>");  // 49406
    eot_token_id_ = encoder_.at("<end_of_text>");     // 49407

    // 8. Seed cache with special tokens
    cache_["<start_of_text>"] = "<start_of_text>";
    cache_["<end_of_text>"]   = "<end_of_text>";
}

// ============================================================================
// BPE algorithm — exact port of Python bpe() method
// ============================================================================

static std::set<std::pair<std::string, std::string>>
get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 1; i < word.size(); ++i) {
        pairs.insert({word[i - 1], word[i]});
    }
    return pairs;
}

std::string SimpleTokenizer::bpe(const std::string& token) const {
    // Check cache
    auto it = cache_.find(token);
    if (it != cache_.end()) {
        return it->second;
    }

    // Build initial word: split into individual chars, add </w> to last
    std::vector<std::string> word;
    // The token is a UTF-8 encoded "byte_encoder" string.
    // Each character in it may be multi-byte UTF-8. We need to split by
    // UTF-8 characters, not bytes.
    {
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = static_cast<unsigned char>(token[i]);
            int len = 1;
            if (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            word.push_back(token.substr(i, len));
            i += len;
        }
    }
    if (word.empty()) {
        return token + "</w>";
    }
    word.back() += "</w>";

    auto pairs = get_pairs(word);
    if (pairs.empty()) {
        std::string result = token + "</w>";
        cache_[token] = result;
        return result;
    }

    while (true) {
        // Find lowest-rank pair
        std::pair<std::string, std::string> best;
        int best_rank = std::numeric_limits<int>::max();
        for (auto& p : pairs) {
            auto r = bpe_ranks_.find(p);
            int rank = (r != bpe_ranks_.end()) ? r->second : std::numeric_limits<int>::max();
            if (rank < best_rank) {
                best_rank = rank;
                best = p;
            }
        }

        if (bpe_ranks_.find(best) == bpe_ranks_.end()) {
            break;  // No more merges possible
        }

        // Merge all occurrences of (best.first, best.second) in word
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            // Find next occurrence of first element
            size_t j = i;
            bool found = false;
            for (; j < word.size(); ++j) {
                if (word[j] == best.first) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                for (; i < word.size(); ++i) {
                    new_word.push_back(word[i]);
                }
                break;
            }

            // Add everything before the match
            for (; i < j; ++i) {
                new_word.push_back(word[i]);
            }

            // Try to merge
            if (word[i] == best.first && i + 1 < word.size() && word[i + 1] == best.second) {
                new_word.push_back(best.first + best.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;
        if (word.size() == 1) break;
        pairs = get_pairs(word);
    }

    // Join with spaces
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i > 0) result += " ";
        result += word[i];
    }

    cache_[token] = result;
    return result;
}

// ============================================================================
// Text cleaning
// ============================================================================

std::string SimpleTokenizer::clean_text(const std::string& text) {
    // Equivalent to: whitespace_clean(basic_clean(text)).lower()
    // Simplified: lowercase + collapse whitespace. Skip ftfy/html unescape
    // since TV prompts are plain ASCII.
    std::string out;
    out.reserve(text.size());
    bool prev_space = false;
    for (char c : text) {
        char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space && !out.empty()) {
                out += ' ';
                prev_space = true;
            }
        } else {
            out += lc;
            prev_space = false;
        }
    }
    // Trim trailing space
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

// ============================================================================
// Regex tokenization (simplified for ASCII)
// ============================================================================

std::vector<std::string> SimpleTokenizer::regex_tokenize(const std::string& text) {
    // Matches the Python regex:
    //   r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
    // Simplified: contractions, letter runs, single digits, other-char runs
    std::vector<std::string> tokens;
    size_t i = 0;

    while (i < text.size()) {
        char c = text[i];

        // Skip whitespace
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }

        // Check contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if (c == '\'' && i + 1 < text.size()) {
            std::string rest = text.substr(i, 4);
            if (rest.size() >= 3 && (rest.substr(0, 3) == "'re" || rest.substr(0, 3) == "'ve" || rest.substr(0, 3) == "'ll")) {
                tokens.push_back(text.substr(i, 3));
                i += 3;
                continue;
            }
            if (rest.size() >= 2 && (rest.substr(0, 2) == "'s" || rest.substr(0, 2) == "'t" ||
                                     rest.substr(0, 2) == "'m" || rest.substr(0, 2) == "'d")) {
                tokens.push_back(text.substr(i, 2));
                i += 2;
                continue;
            }
        }

        // Letter sequence
        if (std::isalpha(static_cast<unsigned char>(c))) {
            size_t start = i;
            while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) ++i;
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        // Single digit
        if (std::isdigit(static_cast<unsigned char>(c))) {
            tokens.push_back(text.substr(i, 1));
            ++i;
            continue;
        }

        // Other non-whitespace, non-letter, non-digit chars
        {
            size_t start = i;
            while (i < text.size() &&
                   !std::isspace(static_cast<unsigned char>(text[i])) &&
                   !std::isalpha(static_cast<unsigned char>(text[i])) &&
                   !std::isdigit(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            if (i > start) {
                tokens.push_back(text.substr(start, i - start));
            }
        }
    }

    return tokens;
}

// ============================================================================
// encode — text → token IDs
// ============================================================================

std::vector<int64_t> SimpleTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> bpe_tokens;
    std::string cleaned = clean_text(text);
    auto raw_tokens = regex_tokenize(cleaned);

    for (auto& token : raw_tokens) {
        // Convert token bytes through byte_encoder
        std::string encoded;
        for (unsigned char byte : token) {
            auto it = byte_encoder_.find(byte);
            if (it != byte_encoder_.end()) {
                encoded += it->second;
            }
        }

        // Apply BPE and look up each sub-token
        std::string bpe_result = bpe(encoded);
        auto sub_tokens = split_string(bpe_result, ' ');
        for (auto& st : sub_tokens) {
            if (st.empty()) continue;
            auto it = encoder_.find(st);
            if (it != encoder_.end()) {
                bpe_tokens.push_back(it->second);
            }
        }
    }

    return bpe_tokens;
}

// ============================================================================
// tokenize — texts → padded [batch, context_length] array
// ============================================================================

std::vector<std::vector<int64_t>>
SimpleTokenizer::tokenize(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int64_t>> result(texts.size(),
                                              std::vector<int64_t>(context_length_, 0));

    for (size_t i = 0; i < texts.size(); ++i) {
        auto ids = encode(texts[i]);

        // Build: [SOT, ...ids..., EOT]
        std::vector<int64_t> tokens;
        tokens.reserve(ids.size() + 2);
        tokens.push_back(sot_token_id_);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
        tokens.push_back(eot_token_id_);

        // Truncate if needed (preserve EOT at end)
        if (static_cast<int>(tokens.size()) > context_length_) {
            tokens.resize(context_length_);
            tokens.back() = eot_token_id_;
        }

        // Copy into padded output (rest stays 0)
        for (size_t j = 0; j < tokens.size(); ++j) {
            result[i][j] = tokens[j];
        }
    }

    return result;
}
