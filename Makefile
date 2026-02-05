CC ?= cc
PKG_CONFIG ?= pkg-config

# Optional manual overrides if pkg-config is not available
FINUFFT_INC ?=
FINUFFT_LIBDIR ?=
GSL_INC ?=
GSL_LIBDIR ?=
FFTW_LIBDIR ?=
OMP_LIBDIR ?=

UNAME_S := $(shell uname -s)

# OpenMP flags (override on the command line if needed)
OMP_CFLAGS ?= -fopenmp
OMP_LDFLAGS ?= -fopenmp
ifeq ($(UNAME_S),Darwin)
  # Homebrew libomp (macOS) uses -lomp
  OMP_LDFLAGS ?= -L$(OMP_LIBDIR) -lomp
endif

CFLAGS = -O3 -fvectorize -march=native \
		 -I./src $(OMP_CFLAGS)
LDFLAGS = -lm $(OMP_LDFLAGS)

# Prefer pkg-config for portability
PKG_CFLAGS := $(shell $(PKG_CONFIG) --cflags finufft fftw3 gsl 2>/dev/null)
PKG_LIBS := $(shell $(PKG_CONFIG) --libs finufft fftw3 gsl 2>/dev/null)

ifneq ($(strip $(PKG_CFLAGS)),)
  CFLAGS += $(PKG_CFLAGS)
else
  CFLAGS += -I$(FINUFFT_INC) -I$(GSL_INC)
endif

ifneq ($(strip $(PKG_LIBS)),)
  LDFLAGS += $(PKG_LIBS)
else
  LDFLAGS += -L$(FINUFFT_LIBDIR) -lfinufft \
		  -L$(FFTW_LIBDIR) -lfftw3 \
		  -L$(GSL_LIBDIR) -lgsl -lgslcblas
endif

SRC = src/test_nufht.c src/fast_hankel_nufht.c src/expansions.c src/bounds.c
OBJ = $(SRC:.c=.o)
EXEC = test_nufht

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

run: $(EXEC)
	./$(EXEC)
