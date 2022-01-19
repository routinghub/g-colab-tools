#!/usr/bin/make

watch:
	ag -l . | entr pip3 install .
