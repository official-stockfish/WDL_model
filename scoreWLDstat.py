import concurrent.futures
import multiprocessing
from collections import Counter
import chess
import chess.pgn
import re
import json
import os
import argparse
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class PosAnalyser:
    def __init__(self, plies):
        self.matching_plies = plies

    def ana_pos(self, files):
        matstats = Counter()
        gameCounter = 0
        for filename in files:
            pgnfilein = open(
                filename, "r", encoding="utf-8-sig", errors="surrogateescape"
            )

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
                    resultkey = {chess.WHITE: "D", chess.BLACK: "D"}
                elif result == "1-0":
                    resultkey = {chess.WHITE: "W", chess.BLACK: "L"}
                elif result == "0-1":
                    resultkey = {chess.WHITE: "L", chess.BLACK: "W"}
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
                        scorekey = score

                    knights = bin(board.knights).count("1")
                    bishops = bin(board.bishops).count("1")
                    rooks = bin(board.rooks).count("1")
                    queens = bin(board.queens).count("1")
                    pawns = bin(board.pawns).count("1")

                    matcountkey = (
                        9 * queens + 5 * rooks + 3 * knights + 3 * bishops + pawns
                    )

                    if scorekey is not None:
                        matstats[
                            (resultkey[turn], plieskey, matcountkey, scorekey)
                        ] += 1

                    board.push(node.move)
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
        "--dir", type=str, default="pgns", help="Directory with the pgns."
    )

    parser.add_argument(
        "--file", type=str, default="", help="A specific file to use."
    )

    args = parser.parse_args()

    if args.file != "":
       pgns = [args.file]
    else:
       pgns = [args.dir + "/" + f for f in os.listdir(args.dir) if f.endswith("pgn")]

    # map sharp_pos to all pgn files using an executor
    ana = PosAnalyser(args.matching_plies)
    targetchunks = 100 * max(1, multiprocessing.cpu_count())
    chunks_size = (len(pgns) + targetchunks - 1) // targetchunks
    pgnschunked = list(chunks(pgns, chunks_size))

    print(
        "Found {} pgn files, creating {} chunks for processing.".format(
            len(pgns), len(pgnschunked)
        )
    )

    res = Counter()
    futures = []

    with tqdm(total=len(pgnschunked), smoothing=0, miniters=1) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as e:
            for entry in pgnschunked:
                futures.append(e.submit(ana.ana_pos, entry))

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                res.update(future.result())

    print("Retained {} scored positions for analysis".format(res.total()))

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
