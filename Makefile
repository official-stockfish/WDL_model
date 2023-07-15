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

all:
	$(CXX) $(CXXFLAGS) $(NATIVE) -o main scoreWLDstat.cpp
