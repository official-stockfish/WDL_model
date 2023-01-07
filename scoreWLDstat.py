import concurrent.futures
from collections import Counter
import chess
import chess.pgn
import re
import json
import os
import argparse

class PosAnalyser:
    def __init__(self, plies):
        self.matching_plies = plies

    def ana_pos(self, filename):
        pgnfilein = open(filename, "r", encoding="utf-8-sig", errors="surrogateescape")
        gameCounter = 0
        matstats = Counter()

        p = re.compile("([+-]*M*[0-9.]*)/([0-9]*)")
        mateRe = re.compile("([+-])M[0-9]*")

        while True:
            # read game
            game = chess.pgn.read_game(pgnfilein)
            if game == None:
                break

            gameCounter = gameCounter + 1

            # get result
            result = game.headers["Result"]
            if result == "1/2-1/2":
                resultkey = "D"
            elif result == "1-0":
                resultkey = "W"
            elif result == "0-1":
                resultkey = "L"
            else:
                continue

            # look at the game,
            plies = 0
            board = game.board()
            for node in game.mainline():

                plies = plies + 1
                if plies > 400:
                    break
                plieskey = (plies + 1) // 2

                turn = board.turn
                scorekey = None
                m = p.search(node.comment)
                if m:
                    score = m.group(1)
                    m = mateRe.search(score)
                    if m:
                        if m.group(1) == "+":
                            score = 1001
                        else:
                            score = -1001
                    else:
                        score = int(float(score) * 100)
                        if score > 1000:
                            score = 1000
                        elif score < -1000:
                            score = -1000
                        score = (score // 5) * 5  # reduce precision
                    if turn == chess.BLACK:
                        score = -score
                    scorekey = score

                knights = bin(board.knights).count("1")
                bishops = bin(board.bishops).count("1")
                rooks = bin(board.rooks).count("1")
                queens = bin(board.queens).count("1")
                pawns = bin(board.pawns).count("1")

                matcountkey = 9 * queens + 5 * rooks + 3 * knights + 3 * bishops + pawns

                if scorekey is not None:
                    matstats[(resultkey, plieskey, matcountkey, scorekey)] += 1

                board = node.board()

        pgnfilein.close()
        return matstats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--matching_plies",
        type=int,
        default=6,
        help="Number of plies that the material situation needs to be on the board (unused).",
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        default="pgns",
        help="Directory with the pgns."
    )
    
    args = parser.parse_args()

    pgns = [args.dir + "/" + f for f in os.listdir(args.dir) if f.endswith("pgn")]
    
    # map sharp_pos to all pgn files using an executor
    ana = PosAnalyser(args.matching_plies)

    res = Counter()
    with concurrent.futures.ProcessPoolExecutor() as e:
        results = e.map(ana.ana_pos, pgns, chunksize=100)
        for r in results:
            res.update(r)

    # and print all the fens
    with open("scoreWLDstat.json", "w") as outfile:
        json.dump(
            {
                str(k): v
                for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)
            },
            outfile,
            indent=1,
        )
