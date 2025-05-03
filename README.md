# SBB Compiler

Stack-based compiler backend. 

## Language Documentation

### What is SBB?
SBB (Stack-Based Backend) is a stack-based compiler backend designed for programming languages that are themselves stack-based and compiled. It provides low-level operations for stack manipulation, memory management, and control flow, intended as a target language for compilers.

### Syntax and Language Features

#### Comments
Single-line comments start with `//`.
```sbb
// This is a comment
```

#### Data Definition
Define data sections using the `data` keyword, followed by the data label (prefixed with `&`), the type (`byte` or `int`), optional size (for byte arrays), and value.
```sbb
data &label_name = byte[size] "value"  // String data
data &number = int 42                // Integer data
```
Example:
```sbb
data &fmt = byte[14] "Hello, world!\n"
data &answer = int 42
```

#### Function Definition
Functions are defined using `set @function_name(type arg1, type arg2) return_type { ... }` where type can be `int` or `byte`.
```sbb
set @function_name(int param1, byte param2) int {
    // function body
}
```
The special function `@main` serves as the entry point of the program.

#### Stack Operations
- `push value`: Pushes a value onto the stack.
- `pop`: Removes the top element from the stack.
- `swap`: Swaps the top two elements.
- `dup`: Duplicates the top element.
- `over`: Copies the second element to the top.

#### Variables
Declare and assign variables using `var $variable_name`.
```sbb
push value
var $my_variable
```

#### Arithmetic Operations
- `add`: Pops two values, adds them, and pushes the result.
- `sub`: Pops two values, subtracts the top from the second, and pushes the result.

#### Control Flow
- Labels: `%label:`
- Jumps: `jmp %label`, `jmp_t %label` (conditional).

#### Function Calls
```sbb
push arg1
push arg2
call @my_function(2)
```

#### Return
```sbb
push result_value
ret int
```

#### Memory Operations
- `remt &data_label`: Pops a value and modifies data at the label.
- `popt &data_label`: Pops a value and writes it to the data label.

### How to Create Programs
1. Define data using `data` with either `byte` or `int` types.
2. Define functions with typed parameters (`int` or `byte`), including `@main`.
3. Use stack operations, variables, arithmetic, and control flow.
4. Call functions with `call`.
5. Return from functions with `ret`.

For examples, see the `examples/` directory. 