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

SRC_FILE = scoreWDLstat.cpp
EXE_FILE = scoreWDLstat
HEADERS = external/chess.hpp external/json.hpp external/threadpool.hpp scoreWDLstat.hpp

all: $(EXE_FILE)

$(EXE_FILE): $(SRC_FILE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(NATIVE) -o $(EXE_FILE) $(SRC_FILE) -lz

format:
	clang-format -i $(SRC_FILE)

clean:
	rm -f $(EXE_FILE) $(EXE_FILE).exe
