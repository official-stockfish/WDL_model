#include "scoreWDLstat.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "external/chess.hpp"
#include "external/gzip/gzstream.h"
#include "external/parallel_hashmap/phmap.h"
#include "external/threadpool.hpp"

namespace fs = std::filesystem;
using json   = nlohmann::json;

using namespace chess;

// unordered map to count (outcome, move, material, score) tuples in pgns
using map_t =
    phmap::parallel_flat_hash_map<Key, int, std::hash<Key>, std::equal_to<Key>,
                                  std::allocator<std::pair<const Key, int>>, 8, std::mutex>;

map_t pos_map = {};

// map to collect metadata for tests
using map_meta = std::unordered_map<std::string, TestMetaData>;

std::atomic<std::size_t> total_chunks = 0;

namespace analysis {

/// @brief Magic value for fishtest pgns, ~1.2 million keys
static constexpr int map_size = 1200000;

/// @brief Analyze a file with pgn games and update the position map, apply filter if present
class Analyze : public pgn::Visitor {
   public:
    Analyze(const std::string &regex_engine, const std::string &move_counter)
        : regex_engine(regex_engine), move_counter(move_counter) {}

    virtual ~Analyze() {}

    void startPgn() override {}

    void startMoves() override {
        do_filter = !regex_engine.empty();

        if (do_filter) {
            if (white.empty() || black.empty()) {
                return;
            }

            std::regex regex(regex_engine);

            if (std::regex_match(white, regex)) {
                filter_side = Color::WHITE;
            }

            if (std::regex_match(black, regex)) {
                if (filter_side == Color::NONE) {
                    filter_side = Color::BLACK;
                } else {
                    do_filter = false;
                }
            }
        }
    }

    void header(std::string_view key, std::string_view value) override {
        if (key == "FEN") {
            std::regex p("0 1$");

            // revert change by cutechess-cli of move counters in .epd books to "0 1"
            if (!move_counter.empty() && std::regex_search(value.data(), p)) {
                board.setFen(std::regex_replace(value.data(), p, "0 " + move_counter));
            } else {
                board.setFen(value);
            }
        }

        if (key == "Variant" && value == "fischerandom") {
            board.set960(true);
        }

        if (key == "Result") {
            hasResult  = true;
            goodResult = true;

            if (value == "1-0") {
                resultkey.white = Result::WIN;
                resultkey.black = Result::LOSS;
            } else if (value == "0-1") {
                resultkey.white = Result::LOSS;
                resultkey.black = Result::WIN;
            } else if (value == "1/2-1/2") {
                resultkey.white = Result::DRAW;
                resultkey.black = Result::DRAW;
            } else {
                goodResult = false;
            }
        }

        if (key == "Termination") {
            if (value == "time forfeit" || value == "abandoned") {
                goodTermination = false;
            }
        }

        if (key == "White") {
            white = value;
        }

        if (key == "Black") {
            black = value;
        }

        skip = !(hasResult && goodTermination && goodResult);
    }

    void move(std::string_view move, std::string_view comment) override {
        if (skip) {
            return;
        }

        if (board.fullMoveNumber() > 200) {
            return;
        }

        Move m;

        m = uci::parseSanInternal(board, move.data(), moves);

        const size_t delimiter_pos = comment.find('/');

        Key key;
        key.score = 1002;

        if (!do_filter || filter_side == board.sideToMove()) {
            if (delimiter_pos != std::string::npos && comment != "book") {
                const auto match_score = comment.substr(0, delimiter_pos);

                if (match_score[1] == 'M') {
                    if (match_score[0] == '+') {
                        key.score = 1001;
                    } else {
                        key.score = -1001;
                    }

                } else {
                    int score = 100 * fast_stof(match_score.data());

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

            pos_map.lazy_emplace_l(
                std::move(key), [&](map_t::value_type &v) { v.second += 1; },
                [&](const map_t::constructor &ctor) { ctor(std::move(key), 1); });
        }

        board.makeMove(m);
    }

    void endPgn() override {
        board.set960(false);
        board.setFen(STARTPOS);

        goodTermination = true;
        hasResult       = false;
        goodResult      = false;

        filter_side = Color::NONE;

        white.clear();
        black.clear();
    }

   private:
    const std::string &regex_engine;
    const std::string &move_counter;

    Board board;
    Movelist moves;

    bool skip = false;

    bool goodTermination = true;
    bool hasResult       = false;
    bool goodResult      = false;

    bool do_filter    = false;
    Color filter_side = Color::NONE;

    std::string white;
    std::string black;

    ResultKey resultkey;
};

void ana_files(const std::vector<std::string> &files, const std::string &regex_engine,
               const map_meta &meta_map, bool fix_fens) {
    for (const auto &file : files) {
        std::string move_counter;

        if (fix_fens) {
            std::string test_filename = file.substr(0, file.find_last_of('-'));

            if (meta_map.find(test_filename) == meta_map.end()) {
                std::cout << "Error: No metadata for test " << test_filename << std::endl;
                std::exit(1);
            }

            if (meta_map.at(test_filename).book_depth.has_value()) {
                move_counter = std::to_string(meta_map.at(test_filename).book_depth.value() + 1);
            } else {
                if (!meta_map.at(test_filename).book.has_value()) {
                    std::cout << "Error: Missing \"book\" key in metadata for test "
                              << test_filename << std::endl;
                    std::exit(1);
                }

                std::regex p(".epd");

                if (std::regex_search(meta_map.at(test_filename).book.value(), p)) {
                    std::cout << "Error: Missing \"book_depth\" key in metadata for .epd book "
                                 "for test "
                              << test_filename << std::endl;
                    std::exit(1);
                }
            }
        }

        const auto pgn_iterator = [&](std::istream &iss) {
            auto vis = std::make_unique<Analyze>(regex_engine, move_counter);

            pgn::StreamParser parser(iss);

            try {
                parser.readGames(*vis);
            } catch (const std::exception &e) {
                std::cout << "Error when parsing: " << file << std::endl;
                std::cerr << e.what() << '\n';
            }
        };

        if (file.size() >= 3 && file.substr(file.size() - 3) == ".gz") {
            igzstream input(file.c_str());
            pgn_iterator(input);
        } else {
            std::ifstream pgn_stream(file);
            pgn_iterator(pgn_stream);
            pgn_stream.close();
        }
    }
}

}  // namespace analysis

[[nodiscard]] map_meta get_metadata(const std::vector<std::string> &file_list,
                                    bool allow_duplicates) {
    map_meta meta_map;
    std::unordered_map<std::string, std::string> test_map;  // map to check for duplicate tests
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
                std::cout << (allow_duplicates ? "Warning" : "Error")
                          << ": Detected a duplicate of test " << test_id << " in directory "
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

            if (!json_file.is_open()) continue;

            json metadata = json::parse(json_file);

            meta_map[test_filename] = metadata.get<TestMetaData>();
        }
    }
    return meta_map;
}

void filter_files_book(std::vector<std::string> &file_list, const map_meta &meta_map,
                       const std::regex &regex_book, bool invert) {
    const auto pred = [&regex_book, invert, &meta_map](const std::string &pathname) {
        std::string test_filename = pathname.substr(0, pathname.find_last_of('-'));

        // check if metadata and "book" entry exist
        if (meta_map.find(test_filename) != meta_map.end() &&
            meta_map.at(test_filename).book.has_value()) {
            bool match = std::regex_match(meta_map.at(test_filename).book.value(), regex_book);
            return invert ? match : !match;
        }

        // missing metadata or "book" entry can never match
        return true;
    };

    file_list.erase(std::remove_if(file_list.begin(), file_list.end(), pred), file_list.end());
}

void filter_files_revision(std::vector<std::string> &file_list, const map_meta &meta_map,
                           const std::regex &regex_rev) {
    const auto pred = [&regex_rev, &meta_map](const std::string &pathname) {
        std::string test_filename = pathname.substr(0, pathname.find_last_of('-'));

        if (meta_map.find(test_filename) == meta_map.end()) {
            return true;
        }
        if (meta_map.at(test_filename).resolved_base.has_value() &&
            std::regex_match(meta_map.at(test_filename).resolved_base.value(), regex_rev)) {
            return false;
        }
        if (meta_map.at(test_filename).resolved_new.has_value() &&
            std::regex_match(meta_map.at(test_filename).resolved_new.value(), regex_rev)) {
            return false;
        }
        return true;
    };

    file_list.erase(std::remove_if(file_list.begin(), file_list.end(), pred), file_list.end());
}

void filter_files_sprt(std::vector<std::string> &file_list, const map_meta &meta_map) {
    const auto pred = [&meta_map](const std::string &pathname) {
        std::string test_filename = pathname.substr(0, pathname.find_last_of('-'));

        // check if metadata and "sprt" entry exist
        if (meta_map.find(test_filename) != meta_map.end() &&
            meta_map.at(test_filename).sprt.has_value() &&
            meta_map.at(test_filename).sprt.value()) {
            return false;
        }

        return true;
    };

    file_list.erase(std::remove_if(file_list.begin(), file_list.end(), pred), file_list.end());
}

void process(const std::vector<std::string> &files_pgn, const std::string &regex_engine,
             const map_meta &meta_map, bool fix_fens, int concurrency) {
    // Create more chunks than threads to prevent threads from idling.
    int target_chunks = 4 * concurrency;

    auto files_chunked = split_chunks(files_pgn, target_chunks);

    std::cout << "Found " << files_pgn.size() << " .pgn(.gz) files, creating "
              << files_chunked.size() << " chunks for processing." << std::endl;

    // Mutex for progress success
    std::mutex progress_mutex;

    // Create a thread pool
    ThreadPool pool(concurrency);

    // Print progress
    std::cout << "\rProgress: " << total_chunks << "/" << files_chunked.size() << std::flush;

    for (const auto &files : files_chunked) {
        pool.enqueue(
            [&files, &regex_engine, &meta_map, &fix_fens, &progress_mutex, &files_chunked]() {
                analysis::ana_files(files, regex_engine, meta_map, fix_fens);

                total_chunks++;

                // Limit the scope of the lock
                {
                    const std::lock_guard<std::mutex> lock(progress_mutex);

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
/// @param json_filename
void save(const std::string &json_filename) {
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

void print_usage(char const *program_name) {
    std::stringstream ss;

    // clang-format off
    ss << "Usage: " << program_name << " [options]" << "\n";
    ss << "Options:" << "\n";
    ss << "  --file <path>         Path to .pgn(.gz) file" << "\n";
    ss << "  --dir <path>          Path to directory containing .pgn(.gz) files (default: pgns)" << "\n";
    ss << "  -r                    Search for .pgn(.gz) files recursively in subdirectories" << "\n";
    ss << "  --allowDuplicates     Allow duplicate directories for test pgns" << "\n";
    ss << "  --concurrency <N>     Number of concurrent threads to use (default: maximum)" << "\n";
    ss << "  --matchRev <regex>    Filter data based on revision SHA in metadata" << "\n";
    ss << "  --matchEngine <regex> Filter data based on engine name in pgns, defaults to matchRev if given" << "\n";
    ss << "  --matchBook <regex>   Filter data based on book name in metadata" << "\n";
    ss << "  --matchBookInvert     Invert the filter" << "\n";
    ss << "  --SPRTonly            Analyse only pgns from SPRT tests" << "\n";
    ss << "  --fixFEN              Patch move counters lost by cutechess-cli" << "\n";
    ss << "  -o <path>             Path to output json file (default: scoreWDLstat.json)" << "\n";
    ss << "  --help                Print this help message" << "\n";
    // clang-format on

    std::cout << ss.str();
}

/// @brief
/// @param argc
/// @param argv See print_usage() for possible arguments
/// @return
int main(int argc, char const *argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    std::vector<std::string> files_pgn;
    std::string regex_book, regex_rev, regex_engine, json_filename = "scoreWDLstat.json";

    std::vector<std::string>::const_iterator pos;

    int concurrency = std::max(1, int(std::thread::hardware_concurrency()));

    if (std::find(args.begin(), args.end(), "--help") != args.end()) {
        print_usage(argv[0]);
        return 0;
    }

    if (find_argument(args, pos, "--concurrency")) {
        concurrency = std::stoi(*std::next(pos));
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

    // sort to easily check for "duplicate" files, i.e. "foo.pgn.gz" and "foo.pgn"
    std::sort(files_pgn.begin(), files_pgn.end());

    for (size_t i = 1; i < files_pgn.size(); ++i) {
        if (files_pgn[i].find(files_pgn[i - 1]) == 0) {
            std::cout << "Error: \"Duplicate\" files: " << files_pgn[i - 1] << " and "
                      << files_pgn[i] << std::endl;
            std::exit(1);
        }
    }

    std::cout << "Found " << files_pgn.size() << " .pgn(.gz) files in total." << std::endl;

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

    if (find_argument(args, pos, "--matchRev")) {
        regex_rev = *std::next(pos);

        if (!regex_rev.empty()) {
            std::cout << "Filtering pgn files matching revision SHA " << regex_rev << std::endl;
            std::regex regex(regex_rev);
            filter_files_revision(files_pgn, meta_map, regex);
        }
        regex_engine = regex_rev;
    }

    bool fix_fens = find_argument(args, pos, "--fixFEN", true);

    if (find_argument(args, pos, "--matchEngine")) {
        regex_engine = *std::next(pos);
    }

    if (find_argument(args, pos, "-o")) {
        json_filename = *std::next(pos);
    }

    pos_map.reserve(analysis::map_size);

    const auto t0 = std::chrono::high_resolution_clock::now();

    process(files_pgn, regex_engine, meta_map, fix_fens, concurrency);

    const auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nTime taken: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0
              << "s" << std::endl;

    save(json_filename);

    return 0;
}
