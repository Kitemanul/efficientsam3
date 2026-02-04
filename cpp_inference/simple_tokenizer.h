#pragma once
/**
 * SimpleTokenizer — C++ port of CLIP's BPE tokenizer.
 *
 * Equivalent to sam3/sam3/model/tokenizer_ve.py::SimpleTokenizer.
 * Loads a gzip-compressed BPE vocabulary file and encodes text into
 * token IDs compatible with MobileCLIP-S1 / CLIP text encoders.
 *
 * Vocab structure (49408 tokens):
 *   [0..255]       256 base byte-unicode characters
 *   [256..511]     256 word-end variants (base + "</w>")
 *   [512..49405]   48894 BPE merges
 *   49406          <start_of_text>
 *   49407          <end_of_text>
 */

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Custom hash for std::pair<std::string, std::string>
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 32) ^ (h2 >> 32);
    }
};

class SimpleTokenizer {
public:
    /**
     * Construct tokenizer from a gzip-compressed BPE vocabulary file.
     * @param bpe_path   Path to bpe_simple_vocab_16e6.txt.gz
     * @param context_length  Maximum sequence length (default 77, CLIP standard)
     */
    explicit SimpleTokenizer(const std::string& bpe_path, int context_length = 77);

    /**
     * Encode a single text string into token IDs (without SOT/EOT or padding).
     * @param text  Raw input text
     * @return  Vector of token IDs
     */
    std::vector<int64_t> encode(const std::string& text) const;

    /**
     * Tokenize one or more texts into a padded 2D array [batch, context_length].
     * Each sequence is: [SOT, ...tokens..., EOT, 0, 0, ...]
     * Truncated to context_length if too long (EOT preserved at end).
     *
     * @param texts  List of input texts
     * @return  2D vector of shape [texts.size(), context_length_]
     */
    std::vector<std::vector<int64_t>> tokenize(const std::vector<std::string>& texts) const;

    int vocab_size()    const { return static_cast<int>(encoder_.size()); }
    int sot_token_id()  const { return sot_token_id_; }
    int eot_token_id()  const { return eot_token_id_; }
    int context_length() const { return context_length_; }

private:
    // ---- Byte ↔ Unicode mapping (256 entries) ----
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    // ---- BPE merge priority (pair → rank, lower = higher priority) ----
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks_;

    // ---- Vocabulary encoder/decoder ----
    std::unordered_map<std::string, int> encoder_;   // token_str → id
    std::unordered_map<int, std::string> decoder_;   // id → token_str

    // ---- BPE result cache (mutable for const encode) ----
    mutable std::unordered_map<std::string, std::string> cache_;

    int context_length_;
    int sot_token_id_;
    int eot_token_id_;

    // ---- Internal helpers ----

    /** Build the 256-entry byte→unicode mapping (matches Python bytes_to_unicode). */
    static std::unordered_map<uint8_t, std::string> build_bytes_to_unicode();

    /** Apply BPE merges to a single word. Returns space-separated BPE tokens. */
    std::string bpe(const std::string& token) const;

    /** Lowercase + collapse whitespace (matches Python clean_fn="lower"). */
    static std::string clean_text(const std::string& text);

    /**
     * Split text into tokens matching CLIP's regex pattern:
     *   'contractions | letter_sequences | single_digits | other_chars'
     * Simplified for ASCII (sufficient for TV text prompts).
     */
    static std::vector<std::string> regex_tokenize(const std::string& text);

    /** Read a gzip file into a string. */
    static std::string read_gzip_file(const std::string& path);

    /** Split string by delimiter. */
    static std::vector<std::string> split_string(const std::string& s, char delim);
};
