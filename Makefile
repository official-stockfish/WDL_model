CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O3

NATIVE = -march=native

# Detect Windows
ifeq ($(OS), Windows_NT)
	uname_S  := Windows
else
ifeq ($(COMP), MINGW)
	uname_S  := Windows
else
	uname_S := $(shell uname -s)
endif
endif

ifeq ($(uname_S), Darwin)
	NATIVE =	
endif

SRC_FILE = scoreWDLstat.cpp external/gzip/gzstream.cpp
EXE_FILE = scoreWDLstat
HEADERS = external/chess.hpp external/json.hpp external/threadpool.hpp scoreWDLstat.hpp external/gzip/gzstream.h

all: $(EXE_FILE)

$(EXE_FILE): $(SRC_FILE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(NATIVE) -o $(EXE_FILE) $(SRC_FILE) -lz

format:
	clang-format -i $(SRC_FILE)
	black -q download_fishtest_pgns.py scoreWDL.py
	shfmt -w -i 4 updateWDL.sh

clean:
	rm -f $(EXE_FILE) $(EXE_FILE).exe
