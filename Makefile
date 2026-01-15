CC = clang
CFLAGS = -O3 -Xclang -fopenmp -fvectorize -march=native -I./src -I/Users/kamion/SCIENCE/FINUFFT/finufft/include -I/opt/homebrew/Cellar/gsl/2.8/include
LDFLAGS = -L/Users/kamion/SCIENCE/FINUFFT/finufft/lib -lfinufft -lm -L/opt/homebrew/opt/fftw/lib -lfftw3 -L/opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -L/opt/homebrew/opt/libomp/lib -lomp

SRC = src/axifresnel_jan13.c src/fast_hankel_nufht.c src/expansions.c src/bounds.c
OBJ = $(SRC:.c=.o)
EXEC = axifresnel

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

run: $(EXEC)
	./$(EXEC)