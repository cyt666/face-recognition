CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
GCC = g++
SHELL = bash
PRINTF	= printf
######################################

NO_COLOR=\e[0m
OK_COLOR=\e[1;32m
ERR_COLOR=\e[1;31m
WARN_COLOR=\e[1;33m
MESG_COLOR=\e[1;34m
FILE_COLOR=\e[1;37m

OK_STRING="[OK]"
ERR_STRING="[ERRORS]"
WARN_STRING="[WARNINGS]"
OK_FMT="${OK_COLOR}%30s\n${NO_COLOR}"
ERR_FMT="${ERR_COLOR}%30s\n${NO_COLOR}"
WARN_FMT="${WARN_COLOR}%30s\n${NO_COLOR}"
PROJECT_ROOT=./

#possible modes LBPH or EIGEN
MODE=LBPH

.PHONY: train create_csv compile run sync clean

train:  create_csv
	@$(GCC) -ggdb $(CFLAGS) -o train_facerecog train_facerecog.cpp $(LIBS) 2> temp.log || touch temp.err
	@if test -e temp.err; \
	then $(PRINTF) $(ERR_FMT) $(ERR_STRING) && cat temp.log; \
	elif test -s temp.log; \
	then $(PRINTF) $(WARN_FMT) $(WARN_STRING) && cat temp.log; \
	else $(PRINTF) "compiled train_facerecog.cpp successfully"; \
	$(PRINTF) $(OK_FMT) $(OK_STRING); \
	./train_facerecog myfaces.csv $(MODE); \
	fi;
	@rm -f temp.log temp.err

create_csv: ./create_csv.py
	@python create_csv.py myfaces > myfaces.csv

compile: 
	@$(GCC) -ggdb $(CFLAGS) -o facerecog facerecog.cpp $(LIBS) 
	@echo "compiled !!!"

run: $(PROJECT_ROOT)/myfaces.csv_*.yml $(PROJECT_ROOT)/myfaces.csv.dat
	@./facerecog myfaces.csv $(MODE)
	
sync:clean
	@rsync -avz --delete ./ ~/Dropbox/Public/face-recog
	@echo "synced with dropbox"
	
clean:
	@echo "Cleaning"
	@rm -f train_facerecog facerecog myfaces.csv*

