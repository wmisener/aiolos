CXX = g++ -std=c++17
CPPFLAGS =      # put pre-processor settings (-I, -D, etc) here
CXXFLAGS = -Wall -Wextra #-pg or -g put compiler settings here
LDFLAGS = -lm       # put linker settings here
BFLAGS = -I. -O3 -g #-pg or -g

##SRC = main.cpp advection.cpp source.cpp
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
INC = $(SRC:.cpp=.h)

TEST_OBJ = $(subst main.o, test_files/main.o, $(OBJ))



#euler_hllc: $(OBJ)
#	$(CXX) -o $@ $(OBJ) $(CXXFLAGS) $(LDFLAGS) $(BFLAGS)

aiolos: $(OBJ) makefile
	$(CXX) -o $@ $(OBJ) $(CXXFLAGS) $(LDFLAGS)
	#./$@

%.o: %.cpp makefile aiolos.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< $(BFLAGS) -o $@ 

.PHONY: tests

tests: $(TEST_OBJ) makefile aiolos.h
	$(CXX) -o  $@  $(TEST_OBJ) $(CXXFLAGS) $(LDFLAGS)
	./tests > /dev/null

clean:
	rm *.o test_files/*.o
