#!/bin/bash
# Author: farismismar
# April 12, 2019

if [ $# -lt 1 ];
then
	echo "Usage: $0 <python_main_file>"
	echo "Do not include the .py extension"
	exit 2 
fi

# check if file exists
if [ ! -f $1.py ];
then
	echo "File $1.py does not exist.  Aborting."
	exit 3
fi

rm -rf build dist __pycache__
py2applet --make-setup $1.py
python3 setup.py py2app -A
#nohup `pwd`/dist/$1.app/Contents/MacOS/$1 1>/dev/null &
`pwd`/dist/$1.app/Contents/MacOS/$1


