# Makefile for FixedNeighborSearch
# This Makefile helps build and install the CUDA extension

# Python executable
PYTHON = python3

# Default target directory
INSTALL_DIR = .

# CUDA paths - adjust these based on your system setup
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64

# Build flags
BUILD_FLAGS = --verbose

# Default target
all: build

# Build the extension
build:
	$(PYTHON) setup.py build_ext $(BUILD_FLAGS)

# Install the package in development mode
develop:
	$(PYTHON) setup.py develop

# Install the package
install:
	$(PYTHON) setup.py install --prefix=$(INSTALL_DIR)

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f fixed_neighbor_search/*.so
	rm -f _fixedneighborsearch*.so

# Clean everything including compiled extensions
distclean: clean
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.so" -delete

# Run tests
test:
	$(PYTHON) -m unittest discover

# Help message
help:
	@echo "Makefile for FixedNeighborSearch"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the extension (default)"
	@echo "  build      - Build the extension"
	@echo "  develop    - Install in development mode"
	@echo "  install    - Install the package"
	@echo "  clean      - Remove build artifacts"
	@echo "  distclean  - Remove all generated files"
	@echo "  test       - Run tests"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON     - Python executable (default: python3)"
	@echo "  INSTALL_DIR - Installation directory (default: .)"
	@echo "  CUDA_HOME  - CUDA installation directory"
	@echo "  BUILD_FLAGS - Additional build flags"

.PHONY: all build develop install clean distclean test help