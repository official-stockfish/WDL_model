
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "chess.hpp"
#include "json.hpp"
#include "threadpool.hpp"

namespace fs = std::filesystem;

using namespace chess;

enum class Result { WIN, LOSS, DRAW };

struct ResultKey {
    Result white;
    Result black;
};

/// @brief [WLD, plies, score, score] tuple key for the position map
using map_key_t = std::tuple<Result, int, int, int>;

struct key_hash {
    std::size_t operator()(const map_key_t &k) const {
        // golden ratio hashing, thus 0x9e3779b9
        std::uint32_t hash = static_cast<int>(std::get<0>(k));
        hash ^= std::get<1>(k) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::get<2>(k) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::get<3>(k) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct key_equal {
    bool operator()(const map_key_t &v0, const map_key_t &v1) const {
        return std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1) &&
               std::get<2>(v0) == std::get<2>(v1) && std::get<3>(v0) == std::get<3>(v1);
    }
};

/// @brief Custom unordered map type to allow for tuple keys
using map_t = std::unordered_map<map_key_t, int, key_hash, key_equal>;

std::atomic<std::size_t> total_chunks = 0;

namespace analysis {

/// @brief Magic value for fishtest pgns, ~1.2 million keys
static constexpr int map_size = 1200000;

/// @brief Analyze a single game and update the position map
/// @param pos_map
/// @param game
void ana_game(map_t &pos_map, const std::optional<Game> &game) {
    if (game.value().headers().find("Result") == game.value().headers().end()) {
        return;
    }

    const auto result = game.value().headers().at("Result");

    ResultKey key;

    if (result == "1-0") {
        key.white = Result::WIN;
        key.black = Result::LOSS;
    } else if (result == "0-1") {
        key.white = Result::LOSS;
        key.black = Result::WIN;
    } else if (result == "1/2-1/2") {
        key.white = Result::DRAW;
        key.black = Result::DRAW;
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

    int plies = 0;

    for (const auto &move : game.value().moves()) {
        plies++;

        if (plies > 400) {
            break;
        }

        const auto match_score = utils::splitString(move.comment, '/');

        const int plieskey = (plies + 1) / 2;

        int score_key = 0;

        bool found_score = false;

        if (match_score.size() >= 1 && move.comment != "book") {
            found_score = true;

            if (match_score[0][1] == 'M') {
                if (match_score[0][1] == '+') {
                    score_key = 1001;
                } else {
                    score_key = -1001;
                }

            } else {
                const auto score = std::stof(match_score[0]);

                int score_adjusted = score * 100;

                if (score_adjusted > 1000) {
                    score_adjusted = 1000;
                } else if (score_adjusted < -1000) {
                    score_adjusted = -1000;
                }

                score_key = int(std::floor(score_adjusted / 5.0)) * 5;
            }
        }

        const auto knights = builtin::popcount(board.pieces(PieceType::KNIGHT));
        const auto bishops = builtin::popcount(board.pieces(PieceType::BISHOP));
        const auto rooks = builtin::popcount(board.pieces(PieceType::ROOK));
        const auto queens = builtin::popcount(board.pieces(PieceType::QUEEN));
        const auto pawns = builtin::popcount(board.pieces(PieceType::PAWN));

        const int matcountkey = 9 * queens + 5 * rooks + 3 * bishops + 3 * knights + pawns;

        if (found_score) {
            const auto turn = board.sideToMove() == Color::WHITE ? key.white : key.black;

            const auto key = std::make_tuple(turn, plieskey, matcountkey, score_key);
            pos_map[key] += 1;
        }

        board.makeMove(move.move);
    }
}

[[nodiscard]] map_t ana_files(std::vector<std::string> files) {
    map_t pos_map;
    pos_map.reserve(map_size);

    for (const auto &file : files) {
        std::ifstream pgn_file(file);
        std::string line;

        while (true) {
            auto game = pgn::readGame(pgn_file);

            if (!game.has_value()) {
                break;
            }

            ana_game(pos_map, game);
        }

        pgn_file.close();
    }

    return pos_map;
}

}  // namespace analysis

/// @brief Get all files from a directory.
/// @param path
/// @return
[[nodiscard]] std::vector<std::string> get_files(std::string_view path = "./pgns") {
    std::vector<std::string> files;

    for (const auto &entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".pgn") {
            files.push_back(entry.path().string());
        }
    }

    return files;
}

/// @brief Split into successive n-sized chunks from pgns.
/// @param pgns
/// @param target_chunks
/// @return
[[nodiscard]] std::vector<std::vector<std::string>> split_chunks(
    const std::vector<std::string> &pgns, int target_chunks) {
    const int chunks_size = (pgns.size() + target_chunks - 1) / target_chunks;

    auto begin = pgns.begin();
    auto end = pgns.end();

    std::vector<std::vector<std::string>> chunks;

    while (begin != end) {
        auto next =
            std::next(begin, std::min(chunks_size, static_cast<int>(std::distance(begin, end))));
        chunks.push_back(std::vector<std::string>(begin, next));
        begin = next;
    }

    return chunks;
}

std::string convertResultToChar(Result result) {
    switch (result) {
        case Result::WIN:
            return "W";
        case Result::LOSS:
            return "L";
        case Result::DRAW:
            return "D";
        default:
            throw std::runtime_error("Invalid result");
    }
}

/// @brief
/// @param argc
/// @param argv Possible ones are --dir and --file
/// @return
int main(int argc, char const *argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    std::vector<std::string> files_pgn;

    if (std::find(args.begin(), args.end(), "--dir") != args.end()) {
        const auto path = std::find(args.begin(), args.end(), "--dir") + 1;
        files_pgn = get_files(*path);
    } else if (std::find(args.begin(), args.end(), "--file") != args.end()) {
        const auto path = std::find(args.begin(), args.end(), "--file") + 1;
        files_pgn = {*path};
    } else {
        files_pgn = get_files();
    }

    // Create more chunks than threads to avoid threads from ideling.
    int target_chunks = 4 * std::max(1, int(std::thread::hardware_concurrency()));

    std::vector<std::vector<std::string>> files_chunked = split_chunks(files_pgn, target_chunks);

    std::cout << "Found " << files_pgn.size() << " pgn files, creating " << files_chunked.size()
              << " chunks for processing." << std::endl;

    map_t pos_map;
    pos_map.reserve(analysis::map_size);

    // Mutex for pos_map access
    std::mutex map_mutex;

    // Create a thread pool
    ThreadPool pool(std::thread::hardware_concurrency());

    // Print progress
    std::cout << "\rProgress: " << total_chunks << "/" << files_chunked.size() << std::flush;

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (const auto &files : files_chunked) {
        pool.enqueue([&files, &map_mutex, &pos_map, &files_chunked]() {
            const auto map = analysis::ana_files(files);

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

    const auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nTime taken: "
              << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s"
              << std::endl;

    std::uint64_t total = 0;

    nlohmann::json j;

    for (const auto &pair : pos_map) {
        const auto map_key_t = "('" + convertResultToChar(std::get<0>(pair.first)) + "', " +
                               std::to_string(std::get<1>(pair.first)) + ", " +
                               std::to_string(std::get<2>(pair.first)) + ", " +
                               std::to_string(std::get<3>(pair.first)) + ")";
        j[map_key_t] = pair.second;
        total += pair.second;
    }

    // save json to file
    std::ofstream outFile("scoreWLDstat.json");
    outFile << j.dump(2);
    outFile.close();

    std::cout << "Retained " << total << " scored positions for analysis." << std::endl;

    return 0;
}
