GEDLIB_ROOT=../gedlib/

INCLUDES=-I$(GEDLIB_ROOT)/src/env/ \
	 -I$(GEDLIB_ROOT)/ext/boost.1.69.0/ \
	 -I$(GEDLIB_ROOT)/ext/eigen.3.3.4/Eigen/ \
	 -I$(GEDLIB_ROOT)/ext/nomad.3.8.1/src/ \
	 -I$(GEDLIB_ROOT)/ext/nomad.3.8.1/ext/sgtelib/src/ \
	 -I$(GEDLIB_ROOT)/ext/lsape.5/include/ \
	 -I$(GEDLIB_ROOT)/ext/libsvm.3.22/ \
	 -I$(GEDLIB_ROOT)/ext/fann.2.2.0/include/

CFLAGS=$(INCLUDES) #-g
LDFLAGS=-L$(GEDLIB_ROOT)/lib \
	-L$(GEDLIB_ROOT)/ext/nomad.3.8.1/lib/ \
       	-L$(GEDLIB_ROOT)/ext/fann.2.2.0/lib/ \
	-L$(GEDLIB_ROOT)/ext/libsvm.3.22/ \
	-lgxlgedlib \
	-lnomad \
	-ldoublefann \
	-lsvm \
	-pthread

all: parallel_hed

parallel_hed: objects/parallel_hed.o
	g++ $^ -o $@ $(LDFLAGS) 

objects/parallel_hed.o: src/parallel_hed.cpp
	g++ $(CFLAGS) -c $^ -o $@

clean:
	rm objects/parallel_hed.o parallel_hed
