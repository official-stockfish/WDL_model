#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "external/json.hpp"

enum class Result { WIN = 'W', DRAW = 'D', LOSS = 'L' };

struct ResultKey {
    Result white;
    Result black;
};

struct Key {
    Result result;             // game result from PoV of side to move
    int move, material, eval;  // move number, material count, engine's eval
    bool operator==(const Key &k) const {
        return result == k.result && move == k.move && material == k.material && eval == k.eval;
    }
    operator std::size_t() const {
        // golden ratio hashing, thus 0x9e3779b9
        std::uint32_t hash = static_cast<int>(result);
        hash ^= move + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= material + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= eval + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
    operator std::string() const {
        return "('" + std::string(1, static_cast<char>(result)) + "', " + std::to_string(move) +
               ", " + std::to_string(material) + ", " + std::to_string(eval) + ")";
    }
};

// overload the std::hash function for Key
template <>
struct std::hash<Key> {
    std::size_t operator()(const Key &k) const { return static_cast<std::size_t>(k); }
};

// overload the std::equal_to function for Key
template <>
struct std::equal_to<Key> {
    bool operator()(const Key &lhs, const Key &rhs) const { return lhs == rhs; }
};

struct TestMetaData {
    std::optional<std::string> book, resolved_base, resolved_new;
    std::optional<bool> sprt;
};

template <typename T = std::string>
std::optional<T> get_optional(const nlohmann::json &j, const char *name) {
    const auto it = j.find(name);
    if (it != j.end()) {
        return std::optional<T>(j[name]);
    } else {
        return std::nullopt;
    }
}

void from_json(const nlohmann::json &nlohmann_json_j, TestMetaData &nlohmann_json_t) {
    auto &j = nlohmann_json_j["args"];

    nlohmann_json_t.sprt = j.contains("sprt") ? std::optional<bool>(true) : std::nullopt;

    nlohmann_json_t.book          = get_optional(j, "book");
    nlohmann_json_t.resolved_base = get_optional(j, "resolved_base");
    nlohmann_json_t.resolved_new  = get_optional(j, "resolved_new");
}

/// @brief Custom stof implementation to avoid locale issues, once clang supports std::from_chars
/// for floats this can be removed
/// @param str
/// @return
inline float fast_stof(const char *str) {
    float result   = 0.0f;
    int sign       = 1;
    int decimal    = 0;
    float fraction = 1.0f;

    // Handle sign
    if (*str == '-') {
        sign = -1;
        str++;
    } else if (*str == '+') {
        str++;
    }

    // Convert integer part
    while (*str >= '0' && *str <= '9') {
        result = result * 10.0f + (*str - '0');
        str++;
    }

    // Convert decimal part
    if (*str == '.') {
        str++;
        while (*str >= '0' && *str <= '9') {
            result = result * 10.0f + (*str - '0');
            fraction *= 10.0f;
            str++;
        }
        decimal = 1;
    }

    // Apply sign and adjust for decimal
    result *= sign;
    if (decimal) {
        result /= fraction;
    }

    return result;
}

/// @brief Get all files from a directory.
/// @param path
/// @param recursive
/// @return
[[nodiscard]] inline std::vector<std::string> get_files(const std::string &path,
                                                        bool recursive = false) {
    std::vector<std::string> files;

    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry)) {
            std::string stem      = entry.path().stem().string();
            std::string extension = entry.path().extension().string();
            if (extension == ".gz") {
                if (stem.size() >= 4 && stem.substr(stem.size() - 4) == ".pgn") {
                    files.push_back(entry.path().string());
                }
            } else if (extension == ".pgn") {
                files.push_back(entry.path().string());
            }
        } else if (recursive && std::filesystem::is_directory(entry)) {
            auto subdir_files = get_files(entry.path().string(), true);
            files.insert(files.end(), subdir_files.begin(), subdir_files.end());
        }
    }

    return files;
}

/// @brief Split into successive n-sized chunks from pgns.
/// @param pgns
/// @param target_chunks
/// @return
[[nodiscard]] inline std::vector<std::vector<std::string>> split_chunks(
    const std::vector<std::string> &pgns, int target_chunks) {
    const int chunks_size = (pgns.size() + target_chunks - 1) / target_chunks;

    auto begin = pgns.begin();
    auto end   = pgns.end();

    std::vector<std::vector<std::string>> chunks;

    while (begin != end) {
        auto next =
            std::next(begin, std::min(chunks_size, static_cast<int>(std::distance(begin, end))));
        chunks.push_back(std::vector<std::string>(begin, next));
        begin = next;
    }

    return chunks;
}

inline bool find_argument(const std::vector<std::string> &args,
                          std::vector<std::string>::const_iterator &pos, std::string_view arg,
                          bool without_parameter = false) {
    pos = std::find(args.begin(), args.end(), arg);

    return pos != args.end() && (without_parameter || std::next(pos) != args.end());
}
