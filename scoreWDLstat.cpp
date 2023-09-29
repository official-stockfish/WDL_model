#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "external/chess.hpp"
#include "external/json.hpp"
#include "external/threadpool.hpp"

namespace fs = std::filesystem;
using json   = nlohmann::json;

using namespace chess;

enum class Result { WIN = 'W', DRAW = 'D', LOSS = 'L' };

struct ResultKey {
    Result white;
    Result black;
};

struct Key {
    Result outcome;             // game outcome from PoV of side to move
    int move, material, score;  // move number, material count, engine's eval
    bool operator==(const Key &k) const {
        return outcome == k.outcome && move == k.move && material == k.material && score == k.score;
    }
    operator std::size_t() const {
        // golden ratio hashing, thus 0x9e3779b9
        std::uint32_t hash = static_cast<int>(outcome);
        hash ^= move + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= material + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= score + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
    operator std::string() const {
        return "('" + std::string(1, static_cast<char>(outcome)) + "', " + std::to_string(move) +
               ", " + std::to_string(material) + ", " + std::to_string(score) + ")";
    }
};

// overload the std::hash function for Key
template <>
struct std::hash<Key> {
    std::size_t operator()(const Key &k) const { return static_cast<std::size_t>(k); }
};

// unordered map to count (outcome, move, material, score) tuples in pgns
using map_t = std::unordered_map<Key, int>;

std::atomic<std::size_t> total_chunks = 0;

namespace analysis {

/// @brief Custom stof implementation to avoid locale issues, once clang supports std::from_chars
/// for floats this can be removed
/// @param str
/// @return
float fast_stof(const char *str) {
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

/// @brief Magic value for fishtest pgns, ~1.2 million keys
static constexpr int map_size = 1200000;

/// @brief Analyze a single game and update the position map, apply filter if present
/// @param pos_map
/// @param game
/// @param regex_engine
void ana_game(map_t &pos_map, const std::optional<Game> &game, const std::string &regex_engine) {
    if (game.value().headers().find("Result") == game.value().headers().end()) {
        return;
    }

    bool do_filter    = !regex_engine.empty();
    Color filter_side = Color::NONE;
    if (do_filter) {
        if (game.value().headers().find("White") == game.value().headers().end() ||
            game.value().headers().find("Black") == game.value().headers().end()) {
            return;
        }
        std::regex regex(regex_engine);
        if (std::regex_match(game.value().headers().at("White"), regex)) {
            filter_side = Color::WHITE;
        }
        if (std::regex_match(game.value().headers().at("Black"), regex)) {
            if (filter_side == Color::NONE) {
                filter_side = Color::BLACK;
            } else {
                do_filter = false;
            }
        }
    }

    const auto result = game.value().headers().at("Result");

    ResultKey resultkey;

    if (result == "1-0") {
        resultkey.white = Result::WIN;
        resultkey.black = Result::LOSS;
    } else if (result == "0-1") {
        resultkey.white = Result::LOSS;
        resultkey.black = Result::WIN;
    } else if (result == "1/2-1/2") {
        resultkey.white = Result::DRAW;
        resultkey.black = Result::DRAW;
    } else {
        return;
    }

    Board board = Board();

    if (game.value().headers().find("FEN") != game.value().headers().end()) {
        board.setFen(game.value().headers().at("FEN"));
    }

    if (game.value().headers().find("Variant") != game.value().headers().end() &&
        game.value().headers().at("Variant") == "fischerandom") {
        board.set960(true);
    }

    for (const auto &move : game.value().moves()) {
        if (board.fullMoveNumber() > 200) {
            break;
        }

        const size_t delimiter_pos = move.comment.find('/');

        Key key;
        key.score = 1002;

        if (!do_filter || filter_side == board.sideToMove()) {
            if (delimiter_pos != std::string::npos && move.comment != "book") {
                const auto match_score = move.comment.substr(0, delimiter_pos);

                if (match_score[1] == 'M') {
                    if (match_score[0] == '+') {
                        key.score = 1001;
                    } else {
                        key.score = -1001;
                    }

                } else {
                    int score = 100 * fast_stof(match_score.c_str());

                    if (score > 1000) {
                        score = 1000;
                    } else if (score < -1000) {
                        score = -1000;
                    }

                    key.score = int(std::floor(score / 5.0)) * 5;  // reduce precision
                }
            }
        }

        if (key.score != 1002) {  // a score was found
            key.outcome = board.sideToMove() == Color::WHITE ? resultkey.white : resultkey.black;
            key.move    = board.fullMoveNumber();
            const auto knights = builtin::popcount(board.pieces(PieceType::KNIGHT));
            const auto bishops = builtin::popcount(board.pieces(PieceType::BISHOP));
            const auto rooks   = builtin::popcount(board.pieces(PieceType::ROOK));
            const auto queens  = builtin::popcount(board.pieces(PieceType::QUEEN));
            const auto pawns   = builtin::popcount(board.pieces(PieceType::PAWN));
            key.material       = 9 * queens + 5 * rooks + 3 * bishops + 3 * knights + pawns;
            pos_map[key] += 1;
        }

        board.makeMove(move.move);
    }
}

void ana_files(map_t &map, const std::vector<std::string> &files, const std::string &regex_engine) {
    map.reserve(map_size);

    for (const auto &file : files) {
        std::ifstream pgn_file(file);
        std::string line;

        while (true) {
            auto game = pgn::readGame(pgn_file);

            if (!game.has_value()) {
                break;
            }

            ana_game(map, game, regex_engine);
        }

        pgn_file.close();
    }
}

}  // namespace analysis

/// @brief Get all files from a directory.
/// @param path
/// @param recursive
/// @return
[[nodiscard]] std::vector<std::string> get_files(const std::string &path, bool recursive = false) {
    std::vector<std::string> files;

    for (const auto &entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry)) {
            if (entry.path().extension() == ".pgn") {
                files.push_back(entry.path().string());
            }
        } else if (recursive && fs::is_directory(entry)) {
            auto subdir_files = get_files(entry.path().string(), true);
            files.insert(files.end(), subdir_files.begin(), subdir_files.end());
        }
    }

    return files;
}

// map to collect metadata for tests
using map_meta = std::map<std::string, json>;

[[nodiscard]] map_meta get_metadata(const std::vector<std::string> &file_list,
                                    bool allow_duplicates) {
    map_meta meta_map;
    std::map<std::string, std::string> test_map;  // map to check for duplicate tests
    std::set<std::string> test_warned;
    for (const auto &pathname : file_list) {
        fs::path path(pathname);
        std::string directory     = path.parent_path().string();
        std::string filename      = path.filename().string();
        std::string test_id       = filename.substr(0, filename.find_last_of('-'));
        std::string test_filename = pathname.substr(0, pathname.find_last_of('-'));

        if (test_map.find(test_id) == test_map.end()) {
            test_map[test_id] = test_filename;
        } else if (test_map[test_id] != test_filename) {
            if (test_warned.find(test_filename) == test_warned.end()) {
                std::cout << (allow_duplicates ? "Warning," : "Error:")
                          << " detected a duplicate of test " << test_id << " in directory "
                          << directory << std::endl;
                test_warned.insert(test_filename);
                if (!allow_duplicates) {
                    std::cout << "Use --allowDuplicates to continue nonetheless." << std::endl;
                    std::exit(1);
                }
            }
        }

        // load the JSON data from disk, only once for each test
        if (meta_map.find(test_filename) == meta_map.end()) {
            std::ifstream json_file(test_filename + ".json");
            if (json_file.is_open()) {
                json metadata;
                json_file >> metadata;
                json_file.close();
                meta_map[test_filename] = metadata;
            }
        }
    }
    return meta_map;
}

void filter_files_book(std::vector<std::string> &file_list, const map_meta &meta_map,
                       const std::regex &regex_book, bool invert) {
    file_list.erase(std::remove_if(file_list.begin(), file_list.end(),
                                   [&regex_book, invert, &meta_map](const std::string &pathname) {
                                       std::string test_filename =
                                           pathname.substr(0, pathname.find_last_of('-'));

                                       // check if metadata and "book" entry exist
                                       if (meta_map.find(test_filename) != meta_map.end() &&
                                           meta_map.at(test_filename).find("book") !=
                                               meta_map.at(test_filename).end()) {
                                           std::string book = meta_map.at(test_filename)["book"];
                                           bool match       = std::regex_match(book, regex_book);
                                           return invert ? match : !match;
                                       }

                                       // missing metadata or "book" entry can never match
                                       return true;
                                   }),
                    file_list.end());
}

void filter_files_sprt(std::vector<std::string> &file_list, const map_meta &meta_map) {
    file_list.erase(std::remove_if(file_list.begin(), file_list.end(),
                                   [&meta_map](const std::string &pathname) {
                                       std::string test_filename =
                                           pathname.substr(0, pathname.find_last_of('-'));

                                       // check if metadata and "sprt" entry exist
                                       if (meta_map.find(test_filename) != meta_map.end() &&
                                           meta_map.at(test_filename).find("sprt") !=
                                               meta_map.at(test_filename).end()) {
                                           return false;
                                       }
                                       return true;
                                   }),
                    file_list.end());
}

/// @brief Split into successive n-sized chunks from pgns.
/// @param pgns
/// @param target_chunks
/// @return
[[nodiscard]] std::vector<std::vector<std::string>> split_chunks(
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

void process(const std::vector<std::string> &files_pgn, map_t &pos_map,
             const std::string &regex_engine) {
    // Create more chunks than threads to prevent threads from idling.
    int target_chunks = 4 * std::max(1, int(std::thread::hardware_concurrency()));

    auto files_chunked = split_chunks(files_pgn, target_chunks);

    std::cout << "Found " << files_pgn.size() << " pgn files, creating " << files_chunked.size()
              << " chunks for processing." << std::endl;

    // Mutex for pos_map access
    std::mutex map_mutex;

    // Create a thread pool
    ThreadPool pool(std::thread::hardware_concurrency());

    // Print progress
    std::cout << "\rProgress: " << total_chunks << "/" << files_chunked.size() << std::flush;

    for (const auto &files : files_chunked) {
        pool.enqueue([&files, &regex_engine, &map_mutex, &pos_map, &files_chunked]() {
            map_t map;
            analysis::ana_files(map, files, regex_engine);

            total_chunks++;

            // Limit the scope of the lock
            {
                const std::lock_guard<std::mutex> lock(map_mutex);

                for (const auto &pair : map) {
                    pos_map[pair.first] += pair.second;
                }

                // Print progress
                std::cout << "\rProgress: " << total_chunks << "/" << files_chunked.size()
                          << std::flush;
            }
        });
    }

    // Wait for all threads to finish
    pool.wait();
}

/// @brief Save the position map to a json file.
/// @param pos_map
/// @param json_filename
void save(const map_t &pos_map, const std::string &json_filename) {
    std::uint64_t total = 0;

    json j;

    for (const auto &pair : pos_map) {
        const auto map_key_t = static_cast<std::string>(pair.first);
        j[map_key_t]         = pair.second;
        total += pair.second;
    }

    // save json to file
    std::ofstream out_file(json_filename);
    out_file << j.dump(2);
    out_file.close();

    std::cout << "Wrote " << total << " scored positions to " << json_filename << " for analysis."
              << std::endl;
}

bool find_argument(const std::vector<std::string> &args,
                   std::vector<std::string>::const_iterator &pos, std::string_view arg,
                   bool without_parameter = false) {
    pos = std::find(args.begin(), args.end(), arg);

    return pos != args.end() && (without_parameter || std::next(pos) != args.end());
}

void print_usage(char const *program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --file <path>         Path to pgn file" << std::endl;
    std::cout << "  --dir <path>          Path to directory containing pgns" << std::endl;
    std::cout << "  -r                    Search for pgns recursively in subdirectories"
              << std::endl;
    std::cout << "  --allowDuplicates     Allow duplicate directories for test pgns" << std::endl;
    std::cout << "  --matchEngine <regex> Filter data based on engine name" << std::endl;
    std::cout << "  --matchBook <regex>   Filter data based on book name" << std::endl;
    std::cout << "  --matchBookInvert     Invert the filter" << std::endl;
    std::cout << "  --SPRTonly            Analyse only pgns from SPRT tests" << std::endl;
    std::cout << "  -o <path>             Path to output json file" << std::endl;
}

/// @brief
/// @param argc
/// @param argv See print_usage() for possible arguments
/// @return
int main(int argc, char const *argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    std::vector<std::string> files_pgn;
    std::string regex_engine, regex_book, json_filename = "scoreWDLstat.json";

    std::vector<std::string>::const_iterator pos;

    if (std::find(args.begin(), args.end(), "--help") != args.end()) {
        print_usage(argv[0]);
        return 0;
    }

    if (find_argument(args, pos, "--file")) {
        files_pgn = {*std::next(pos)};
    } else {
        std::string path = "./pgns";
        if (find_argument(args, pos, "--dir")) {
            path = *std::next(pos);
        }
        bool recursive = find_argument(args, pos, "-r", true);
        std::cout << "Looking " << (recursive ? "(recursively) " : "") << "for pgn files in "
                  << path << std::endl;

        files_pgn = get_files(path, recursive);
    }

    bool allow_duplicates = find_argument(args, pos, "--allowDuplicates", true);
    auto meta_map         = get_metadata(files_pgn, allow_duplicates);

    if (find_argument(args, pos, "--SPRTonly", true)) {
        filter_files_sprt(files_pgn, meta_map);
    }

    if (find_argument(args, pos, "--matchBook")) {
        regex_book = *std::next(pos);
        if (!regex_book.empty()) {
            bool invert = find_argument(args, pos, "--matchBookInvert", true);
            std::cout << "Filtering pgn files " << (invert ? "not " : "")
                      << "matching the book name " << regex_book << std::endl;
            std::regex regex(regex_book);
            filter_files_book(files_pgn, meta_map, regex, invert);
        }
    }

    if (find_argument(args, pos, "--matchEngine")) {
        regex_engine = *std::next(pos);
    }

    if (find_argument(args, pos, "-o")) {
        json_filename = *std::next(pos);
    }

    map_t pos_map;
    pos_map.reserve(analysis::map_size);

    const auto t0 = std::chrono::high_resolution_clock::now();

    process(files_pgn, pos_map, regex_engine);

    const auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nTime taken: "
              << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s"
              << std::endl;

    save(pos_map, json_filename);

    return 0;
}
