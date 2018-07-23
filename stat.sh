#!/bin/bash
awk -F, 'END {printf "Number of Rows : %s\nNumber of Columns = %s\n", NR, NF}' $1
