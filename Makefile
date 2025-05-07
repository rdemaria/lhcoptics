.PHONY: docs clean

docs:
	sphinx-build -b html doc/source doc/build

clean:
	rm -rf doc/build