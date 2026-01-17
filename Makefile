CC = clang

FINUFFT_INC ?= /Users/kamion/SCIENCE/FINUFFT/finufft/include
FINUFFT_LIBDIR ?= /Users/kamion/SCIENCE/FINUFFT/finufft/lib
GSL_INC ?= /opt/homebrew/Cellar/gsl/2.8/include
GSL_LIBDIR ?= /opt/homebrew/Cellar/gsl/2.8/lib
FFTW_LIBDIR ?= /opt/homebrew/opt/fftw/lib
OMP_LIBDIR ?= /opt/homebrew/opt/libomp/lib

CFLAGS = -O3 -Xclang -fopenmp -fvectorize -march=native \
		 -I./src -I$(FINUFFT_INC) -I$(GSL_INC)
LDFLAGS = -L$(FINUFFT_LIBDIR) -lfinufft -lm \
		  -L$(FFTW_LIBDIR) -lfftw3 \
		  -L$(GSL_LIBDIR) -lgsl -lgslcblas \
		  -L$(OMP_LIBDIR) -lomp

SRC = src/axifresnel.c src/fast_hankel_nufht.c src/expansions.c src/bounds.c
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
