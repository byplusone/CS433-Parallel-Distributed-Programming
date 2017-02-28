GCC=gcc


all:
	$(GCC) -Wall -O2 -o mapper mapper.c
	$(GCC) -Wall -O2 -o reducer reducer.c
clean:
	rm -rf mapper reducer *.dat
run:
	cat input.txt| ./mapper | sort > test_result.dat
