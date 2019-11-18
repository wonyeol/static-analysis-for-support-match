#-------------------------------
# Entry points
all: execs
execs: pyppai.exe batch.exe
#-------------------------------
# Executables
dist:
	obuild configure
pyppai.exe: dist
	obuild build exe-pyppai.exe
batch.exe: dist
	obuild build exe-batch.exe
#-------------------------------
# Running batch
batch-suite: batch.exe
	./batch.exe suite -q -to 10
batch-examples: batch.exe
	./batch.exe examples -q -to 10
#-------------------------------
# PHONY misc targets
.PHONY: pyppai.exe batch.exe \
	execs batch-suite batch-examples wc edit clean
wc:
	wc src/*ml src/*.mli
edit:
	emacs --background-color=Black --foreground-color=White makefile &
clean: 
	rm -rf src/*~ dist pyppai.exe batch.exe

