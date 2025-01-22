#include "scoreWDLstat.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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

// unordered map to count (result, move, material, eval) tuples in pgns
using map_t =
    phmap::parallel_flat_hash_map<Key, int, std::hash<Key>, std::equal_to<Key>,
                                  std::allocator<std::pair<const Key, int>>, 8, std::mutex>;

// map to collect metadata for tests
using map_meta = std::unordered_map<std::string, TestMetaData>;

// map to hold move counters that cutechess-cli changed from original FENs
using map_fens = std::unordered_map<std::string, std::pair<int, int>>;

// concurrent position map
map_t pos_map                         = {};
std::atomic<std::size_t> total_chunks = 0;
std::atomic<std::size_t> total_games  = 0;

namespace analysis {

/// @brief Magic value for fishtest pgns, ~1.2 million keys
static constexpr int map_size = 1200000;

/// @brief Analyze a file with pgn games and update the position map, apply filter if present
class Analyze : public pgn::Visitor {
   public:
    Analyze(const std::string &regex_engine, const map_fens &fixfen_map, const int bin_width)
        : regex_engine(regex_engine), fixfen_map(fixfen_map), bin_width(bin_width) {}

    virtual ~Analyze() {}

    void startPgn() override {}

    void startMoves() override {
        if (!skip) {
            total_games++;
        }

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
            std::regex p("^(.+) 0 1$");
            std::smatch match;
            std::string value_str(value);

            // revert changes by cutechess-cli to move counters
            if (!fixfen_map.empty() && std::regex_search(value_str, match, p) && match.size() > 1) {
                std::string fen = match[1];
                auto it         = fixfen_map.find(fen);

                if (it == fixfen_map.end()) {
                    std::cerr << "Could not find FEN " << fen << " in fixFENsource." << std::endl;
                    std::exit(1);
                }

                const auto &fix = it->second;
                std::string fixed_value =
                    fen + " " + std::to_string(fix.first) + " " + std::to_string(fix.second);
                board.setFen(fixed_value);
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
            if (value == "time forfeit" || value == "abandoned" || value == "stalled connection" ||
                value == "illegal move" || value == "unterminated") {
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

        // openbench uses Nf3 {+0.57 17/28 583 363004}, fishtest Nf3 {+0.57/17}
        const size_t delimiter_pos = comment.find_first_of(" /");

        Key key;
        key.eval = 1002;

        if (!do_filter || filter_side == board.sideToMove()) {
            if (delimiter_pos != std::string::npos && comment != "book") {
                const auto match_eval = comment.substr(0, delimiter_pos);

                if (match_eval[1] == 'M') {
                    if (match_eval[0] == '+') {
                        key.eval = 1001;
                    } else {
                        key.eval = -1001;
                    }

                } else {
                    int eval = 100 * fast_stof(match_eval.data());

                    if (eval > 1000) {
                        eval = 1000;
                    } else if (eval < -1000) {
                        eval = -1000;
                    }

                    // reduce precision
                    key.eval = int(std::round(eval / float(bin_width))) * bin_width;
                }
            }
        }

        // an eval was found
        if (key.eval != 1002) {
            const auto knights = board.pieces(PieceType::KNIGHT).count();
            const auto bishops = board.pieces(PieceType::BISHOP).count();
            const auto rooks   = board.pieces(PieceType::ROOK).count();
            const auto queens  = board.pieces(PieceType::QUEEN).count();
            const auto pawns   = board.pieces(PieceType::PAWN).count();

            key.result   = board.sideToMove() == Color::WHITE ? resultkey.white : resultkey.black;
            key.move     = board.fullMoveNumber();
            key.material = 9 * queens + 5 * rooks + 3 * bishops + 3 * knights + pawns;

            // insert or update the position map
            pos_map.lazy_emplace_l(
                std::move(key), [&](map_t::value_type &v) { v.second += 1; },
                [&](const map_t::constructor &ctor) { ctor(std::move(key), 1); });
        }

        Move m;

        m = uci::parseSan(board, move, moves);

        // chess-lib may call move() with empty strings for move
        if (m == Move::NO_MOVE) {
            this->skipPgn(true);
            return;
        }

        board.makeMove<true>(m);
    }

    void endPgn() override {
        board.set960(false);
        board.setFen(constants::STARTPOS);

        goodTermination = true;
        hasResult       = false;
        goodResult      = false;

        filter_side = Color::NONE;

        white.clear();
        black.clear();
    }

   private:
    const std::string &regex_engine;
    const map_fens &fixfen_map;
    const int bin_width;

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
               const map_fens &fixfen_map, const int bin_width) {
    for (const auto &file : files) {
        const auto pgn_iterator = [&](std::istream &iss) {
            auto vis = std::make_unique<Analyze>(regex_engine, fixfen_map, bin_width);

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

[[nodiscard]] map_fens get_fixfen(std::string file) {
    map_fens fixfen_map;
    if (file.empty()) {
        return fixfen_map;
    }

    const auto fen_iterator = [&](std::istream &iss) {
        std::string line;
        while (std::getline(iss, line)) {
            std::istringstream iss(line);
            std::string f1, f2, f3, ep;
            int halfmove, fullmove = 0;

            iss >> f1 >> f2 >> f3 >> ep >> halfmove >> fullmove;

            if (!fullmove) continue;

            auto key         = f1 + ' ' + f2 + ' ' + f3 + ' ' + ep;
            auto fixfen_data = std::pair<int, int>(halfmove, fullmove);

            if (fixfen_map.find(key) != fixfen_map.end()) {
                // for duplicate FENs, prefer the one with lower full move counter
                if (fullmove < fixfen_map[key].second) {
                    fixfen_map[key] = fixfen_data;
                }
            } else {
                fixfen_map[key] = fixfen_data;
            }
        }
    };

    if (file.size() >= 3 && file.substr(file.size() - 3) == ".gz") {
        igzstream input(file.c_str());
        fen_iterator(input);
    } else {
        std::ifstream input(file);
        fen_iterator(input);
    }

    return fixfen_map;
}

[[nodiscard]] map_meta get_metadata(const std::vector<std::string> &file_list,
                                    bool allow_duplicates) {
    map_meta meta_map;
    // map to check for duplicate tests
    std::unordered_map<std::string, std::string> test_map;
    std::set<std::string> test_warned;

    for (const auto &pathname : file_list) {
        fs::path path(pathname);
        std::string filename      = path.filename().string();
        std::string test_id       = filename.substr(0, filename.find_first_of("-."));
        std::string test_filename = (path.parent_path() / test_id).string();

        if (test_map.find(test_id) == test_map.end()) {
            test_map[test_id] = test_filename;
        } else if (test_map[test_id] != test_filename) {
            if (test_warned.find(test_filename) == test_warned.end()) {
                std::cout << (allow_duplicates ? "Warning" : "Error")
                          << ": Detected a duplicate of test " << test_id << " in directory "
                          << path.parent_path().string() << std::endl;
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

template <typename STRATEGY>
void filter_files(std::vector<std::string> &file_list, const map_meta &meta_map,
                  const STRATEGY &strategy) {
    const auto applier = [&](const std::string &pathname) {
        fs::path path(pathname);
        std::string filename      = path.filename().string();
        std::string test_id       = filename.substr(0, filename.find_first_of("-."));
        std::string test_filename = (path.parent_path() / test_id).string();
        return strategy.apply(test_filename, meta_map);
    };
    const auto it = std::remove_if(file_list.begin(), file_list.end(), applier);
    file_list.erase(it, file_list.end());
}

class BookFilterStrategy {
    std::regex regex_book;
    bool invert;

   public:
    BookFilterStrategy(const std::regex &rb, bool inv) : regex_book(rb), invert(inv) {}

    bool apply(const std::string &filename, const map_meta &meta_map) const {
        // check if metadata and "book" entry exist
        if (meta_map.find(filename) != meta_map.end() && meta_map.at(filename).book.has_value()) {
            bool match = std::regex_match(meta_map.at(filename).book.value(), regex_book);
            return invert ? match : !match;
        }

        // missing metadata or "book" entry can never match
        return true;
    }
};

class RevFilterStrategy {
    std::regex regex_rev;

   public:
    RevFilterStrategy(const std::regex &rb) : regex_rev(rb) {}

    bool apply(const std::string &filename, const map_meta &meta_map) const {
        if (meta_map.find(filename) == meta_map.end()) {
            return true;
        }

        if (meta_map.at(filename).resolved_base.has_value() &&
            std::regex_match(meta_map.at(filename).resolved_base.value(), regex_rev)) {
            return false;
        }

        if (meta_map.at(filename).resolved_new.has_value() &&
            std::regex_match(meta_map.at(filename).resolved_new.value(), regex_rev)) {
            return false;
        }

        return true;
    }
};

class TcFilterStrategy {
    std::regex regex_tc;

   public:
    TcFilterStrategy(const std::regex &rb) : regex_tc(rb) {}

    bool apply(const std::string &filename, const map_meta &meta_map) const {
        if (meta_map.find(filename) == meta_map.end()) {
            return true;
        }

        if (meta_map.at(filename).new_tc.has_value() && meta_map.at(filename).tc.has_value()) {
            if (meta_map.at(filename).new_tc.value() != meta_map.at(filename).tc.value()) {
                return true;
            }

            if (std::regex_match(meta_map.at(filename).tc.value(), regex_tc)) {
                return false;
            }
        }

        return true;
    }
};

class ThreadsFilterStrategy {
    int threads;

   public:
    ThreadsFilterStrategy(int t) : threads(t) {}

    bool apply(const std::string &filename, const map_meta &meta_map) const {
        if (meta_map.find(filename) == meta_map.end()) {
            return true;
        }

        if (meta_map.at(filename).threads.has_value() &&
            meta_map.at(filename).threads.value() == threads) {
            return false;
        }

        return true;
    }
};

class EloFilterStrategy {
    double EloDiffMin, EloDiffMax;

   public:
    EloFilterStrategy(double mi, double ma) : EloDiffMin(mi), EloDiffMax(ma) {}

    double pentanomialToEloDiff(const std::vector<int> &pentanomial) const {
        auto pairs            = std::accumulate(pentanomial.begin(), pentanomial.end(), 0);
        const double WW       = double(pentanomial[4]) / pairs;
        const double WD       = double(pentanomial[3]) / pairs;
        const double WLDD     = double(pentanomial[2]) / pairs;
        const double LD       = double(pentanomial[1]) / pairs;
        const double LL       = double(pentanomial[0]) / pairs;
        const double score    = WW + 0.75 * WD + 0.5 * WLDD + 0.25 * LD;
        const double WW_dev   = WW * std::pow((1 - score), 2);
        const double WD_dev   = WD * std::pow((0.75 - score), 2);
        const double WLDD_dev = WLDD * std::pow((0.5 - score), 2);
        const double LD_dev   = LD * std::pow((0.25 - score), 2);
        const double LL_dev   = LL * std::pow((0 - score), 2);
        const double variance = WW_dev + WD_dev + WLDD_dev + LD_dev + LL_dev;
        return (score - 0.5) / std::sqrt(2 * variance) * (800 / std::log(10));
    }

    bool apply(const std::string &filename, const map_meta &meta_map) const {
        if (meta_map.find(filename) == meta_map.end()) {
            return true;
        }

        if (!meta_map.at(filename).pentanomial.has_value()) {
            return true;
        }

        double fileEloDiff = pentanomialToEloDiff(meta_map.at(filename).pentanomial.value());
        if (EloDiffMin <= fileEloDiff && fileEloDiff <= EloDiffMax) {
            return false;
        }

        return true;
    }
};

class SprtFilterStrategy {
   public:
    bool apply(const std::string &filename, const map_meta &meta_map) const {
        // check if metadata and "sprt" entry exist
        if (meta_map.find(filename) != meta_map.end() && meta_map.at(filename).sprt.has_value() &&
            meta_map.at(filename).sprt.value()) {
            return false;
        }

        return true;
    }
};

void process(const std::vector<std::string> &files_pgn, const std::string &regex_engine,
             const map_fens &fixfen_map, int concurrency, int bin_width) {
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
            [&files, &regex_engine, &fixfen_map, &progress_mutex, &files_chunked, &bin_width]() {
                analysis::ana_files(files, regex_engine, fixfen_map, bin_width);

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
    std::uint64_t total_pos = 0;

    json j;

    for (const auto &pair : pos_map) {
        const auto map_key_t = static_cast<std::string>(pair.first);
        j[map_key_t]         = pair.second;
        total_pos += pair.second;
    }

    // save json to file
    std::ofstream out_file(json_filename);
    out_file << j.dump(2);
    out_file.close();

    std::cout << "Wrote " << total_pos << " scored positions from " << total_games << " games to "
              << json_filename << " for analysis." << std::endl;
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
    ss << "  --matchTC <regex>     Filter data based on time control in metadata" << "\n";
    ss << "  --matchThreads <N>    Filter data based on used threads in metadata" << "\n";
    ss << "  --matchBook <regex>   Filter data based on book name in metadata" << "\n";
    ss << "  --matchBookInvert     Invert the filter" << "\n";
    ss << "  --EloDiffMax <X>      Filter data based on estimated nElo difference" << "\n";
    ss << "  --EloDiffMin <Y>      Filter data based on estimated nElo difference (defaults to -X if X is given)" << "\n";
    ss << "  --SPRTonly            Analyse only pgns from SPRT tests" << "\n";
    ss << "  --fixFENsource        Patch move counters lost by cutechess-cli based on FENs in this file" << "\n";
    ss << "  --binWidth            bin position scores for faster processing and smoother densities (default 5)" << "\n";
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
    // Workaround to prevent data races in std::ctype<char>::narrow
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77704
#if __GLIBCXX__
    {
        const std::ctype<char> &ct(std::use_facet<std::ctype<char>>(std::locale()));

        for (size_t i(0); i != 256; ++i) ct.narrow(static_cast<char>(i), '\0');
    }
#endif

    pos_map.reserve(analysis::map_size);

    CommandLine cmd(argc, argv);

    std::vector<std::string> files_pgn;
    std::string json_filename = "scoreWDLstat.json";
    std::string default_path  = "./pgns";
    std::string regex_engine;
    map_fens fixfen_map;
    int bin_width   = 5;
    int concurrency = std::max(1, int(std::thread::hardware_concurrency()));

    if (cmd.has_argument("--help", true)) {
        print_usage(argv[0]);
        return 0;
    }

    if (cmd.has_argument("--binWidth")) {
        bin_width = std::stoi(cmd.get_argument("--binWidth"));
    }

    if (cmd.has_argument("--concurrency")) {
        concurrency = std::stoi(cmd.get_argument("--concurrency"));
    }

    if (cmd.has_argument("--file")) {
        files_pgn = {cmd.get_argument("--file")};
    } else {
        auto path = cmd.get_argument("--dir", default_path);

        bool recursive = cmd.has_argument("-r", true);
        std::cout << "Looking " << (recursive ? "(recursively) " : "") << "for pgn files in "
                  << path << std::endl;

        files_pgn = get_files(path, recursive);

        // sort to easily check for "duplicate" files, i.e. "foo.pgn.gz" and "foo.pgn"
        std::sort(files_pgn.begin(), files_pgn.end());

        for (size_t i = 1; i < files_pgn.size(); ++i) {
            if (files_pgn[i].find(files_pgn[i - 1]) == 0) {
                std::cout << "Error: \"Duplicate\" files: " << files_pgn[i - 1] << " and "
                          << files_pgn[i] << std::endl;
                std::exit(1);
            }
        }
    }

    std::cout << "Found " << files_pgn.size() << " .pgn(.gz) files in total." << std::endl;

    auto meta_map = get_metadata(files_pgn, cmd.has_argument("--allowDuplicates", true));

    if (cmd.has_argument("--SPRTonly", true)) {
        filter_files(files_pgn, meta_map, SprtFilterStrategy());
    }

    if (cmd.has_argument("--matchBook")) {
        auto regex_book = cmd.get_argument("--matchBook");

        if (!regex_book.empty()) {
            bool invert = cmd.has_argument("--matchBookInvert", true);
            std::cout << "Filtering pgn files " << (invert ? "not " : "")
                      << "matching the book name " << regex_book << std::endl;
            filter_files(files_pgn, meta_map, BookFilterStrategy(std::regex(regex_book), invert));
        }
    }

    if (cmd.has_argument("--matchRev")) {
        auto regex_rev = cmd.get_argument("--matchRev");

        if (!regex_rev.empty()) {
            std::cout << "Filtering pgn files matching revision SHA " << regex_rev << std::endl;
            filter_files(files_pgn, meta_map, RevFilterStrategy(std::regex(regex_rev)));
        }

        regex_engine = regex_rev;
    }

    if (cmd.has_argument("--matchTC")) {
        auto regex_tc = cmd.get_argument("--matchTC");

        if (!regex_tc.empty()) {
            std::cout << "Filtering pgn files matching TC " << regex_tc << std::endl;
            filter_files(files_pgn, meta_map, TcFilterStrategy(std::regex(regex_tc)));
        }
    }

    if (cmd.has_argument("--matchThreads")) {
        int threads = std::stoi(cmd.get_argument("--matchThreads"));

        std::cout << "Filtering pgn files using threads = " << threads << std::endl;
        filter_files(files_pgn, meta_map, ThreadsFilterStrategy(threads));
    }

    if (cmd.has_argument("--EloDiffMax") || cmd.has_argument("--EloDiffMin")) {
        double ma = std::numeric_limits<double>::infinity();
        if (cmd.has_argument("--EloDiffMax")) {
            ma = std::stod(cmd.get_argument("--EloDiffMax"));
        }
        double mi = -ma;
        if (cmd.has_argument("--EloDiffMin")) {
            mi = std::stod(cmd.get_argument("--EloDiffMin"));
        }

        std::cout << "Filtering pgn files with nElo in [" << mi << ", " << ma << "]" << std::endl;
        if (mi != -ma && !cmd.has_argument("--SPRTonly", true)) {
            std::cout << "Warning: Asymmetric nElo window suggests --SPRTonly should be used!"
                      << std::endl;
        }

        filter_files(files_pgn, meta_map, EloFilterStrategy(mi, ma));
    }

    if (cmd.has_argument("--fixFENsource")) {
        fixfen_map = get_fixfen(cmd.get_argument("--fixFENsource"));
    }

    if (cmd.has_argument("--matchEngine")) {
        regex_engine = cmd.get_argument("--matchEngine");
    }

    if (cmd.has_argument("-o")) {
        json_filename = cmd.get_argument("-o");
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    process(files_pgn, regex_engine, fixfen_map, concurrency, bin_width);
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nTime taken: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0
              << "s" << std::endl;

    save(json_filename);

    return 0;
}
