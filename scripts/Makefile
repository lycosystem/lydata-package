.PHONY: all old new compare pretty-table-old pretty-table-new

all: old.csv new.csv compare pretty-table-old pretty-table-new

old.csv: old.py
	uv run old.py

new.csv: new.py
	uv run new.py

compare: old.csv new.csv
	uv run compare.py old.csv new.csv

old.diff.csv new.diff.csv: compare
	@true  # compare.py is assumed to produce both diff files

pretty-table-old: old.diff.csv
	uvx tabulate -f "simple" -s "," old.diff.csv > old.pretty.csv

pretty-table-new: new.diff.csv
	uvx tabulate -f "simple" -s "," new.diff.csv > new.pretty.csv

clean:
	rm -f old.csv new.csv old.diff.csv new.diff.csv old.pretty.csv new.pretty.csv
