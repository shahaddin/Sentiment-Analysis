# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++11 -Wall -Wextra -O3 --verbose

# Include directories
INCLUDES := -I include

# Source directory
SRC_DIR := src

# Object directory
OBJ_DIR := obj

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Output executable
TARGET := ForwardPass

# Build rule for the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# Build rule for object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Dependency rule for main.o
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(wildcard $(SRC_DIR)/*.h)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	@rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: clean
