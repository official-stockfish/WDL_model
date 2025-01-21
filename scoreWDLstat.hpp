#include <zlib.h>

#include <algorithm>
#include <charconv>
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
    std::optional<std::string> book, new_tc, resolved_base, resolved_new, tc;
    std::optional<int> threads;
    std::optional<bool> sprt;
    std::optional<std::vector<int>> pentanomial;
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
    nlohmann_json_t.new_tc        = get_optional(j, "new_tc");
    nlohmann_json_t.resolved_base = get_optional(j, "resolved_base");
    nlohmann_json_t.resolved_new  = get_optional(j, "resolved_new");
    nlohmann_json_t.tc            = get_optional(j, "tc");
    nlohmann_json_t.threads       = get_optional<int>(j, "threads");

    auto &jr                    = nlohmann_json_j["results"];
    nlohmann_json_t.pentanomial = get_optional<std::vector<int>>(jr, "pentanomial");
}

// #if (defined(__clang__) && __clang_major__ < 20) || (__GLIBCXX__ && __GLIBCXX__ < 20200122)

#if (defined(__clang__) && __clang_major__ < 20) || \
    !(defined(__GNUC__) && (__GNUC__ >= 11 && __GNUC_MINOR__ >= 1))
/// @brief Custom stof implementation to avoid locale issues, once clang supports std::from_chars
/// for floats this can be removed
/// @param str
/// @return
inline float fast_stof(std::string_view str) {
    float result   = 0.0f;
    int sign       = 1;
    int decimal    = 0;
    float fraction = 1.0f;

    const char *ptr = str.data();
    const char *end = ptr + str.size();

    // Handle sign
    if (ptr < end && *ptr == '-') {
        sign = -1;
        ptr++;
    } else if (ptr < end && *ptr == '+') {
        ptr++;
    }

    // Convert integer part
    while (ptr < end && *ptr >= '0' && *ptr <= '9') {
        result = result * 10.0f + (*ptr - '0');
        ptr++;
    }

    // Convert decimal part
    if (ptr < end && *ptr == '.') {
        ptr++;
        while (ptr < end && *ptr >= '0' && *ptr <= '9') {
            result = result * 10.0f + (*ptr - '0');
            fraction *= 10.0f;
            ptr++;
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
#else
inline float fast_stof(std::string_view sw) {
    if (sw[0] == '+') {
        sw.remove_prefix(1);
    }

    float result;
    const char *str_end  = sw.data() + sw.length();
    const auto fc_result = std::from_chars(sw.data(), str_end, result);

    if (fc_result.ec == std::errc()) {
        return result;
    } else if (fc_result.ec == std::errc::invalid_argument) {
        throw std::invalid_argument("Invalid float format");
    } else if (fc_result.ec == std::errc::result_out_of_range) {
        throw std::out_of_range("Float value out of range");
    }

    throw std::runtime_error("Unknown error in float conversion");
}
#endif

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

class CommandLine {
   public:
    CommandLine(int argc, char const *argv[]) {
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
    }

    bool has_argument(const std::string &arg, bool without_parameter = false) const {
        const auto pos = std::find(args.begin(), args.end(), arg);
        return pos != args.end() && (without_parameter || std::next(pos) != args.end());
    }

    std::string get_argument(const std::string &arg, std::string default_value = "") const {
        auto it = std::find(args.begin(), args.end(), arg);

        if (it != args.end() && std::next(it) != args.end()) {
            return *std::next(it);
        }

        return default_value;
    }

   private:
    std::vector<std::string> args;
};
